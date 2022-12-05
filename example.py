# Author Joey Whelan
from argparse import ArgumentParser
from enum import Enum
from typing import Tuple
from redis import Connection, from_url
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from img2vec_pytorch import Img2Vec
from PIL import Image
import os
import json
import numpy as np
import random
import pandas as pd
from enum_actions import enum_action

IMAGE_DIR: str = 'data/images'               # directory of images
VECTOR_FILE: str = 'data/vectors.json'       # JSON file containing image ids and their embeddings
NUM_IMAGES: int = 100                        # Number of images to be vectorized from the image dir
TOPK: int = 5                                # Number of results to be returned from the VSS query
REDIS_URL: str = 'redis://localhost:6379'  

class OBJECT_TYPE(Enum):
    HASH = 'hash'
    JSON = 'json'

class INDEX_TYPE(Enum):
    FLAT = 'flat'
    HNSW = 'hnsw'

class METRIC_TYPE(Enum):
    L2 = 'l2'
    IP = 'ip'
    COSINE = 'cosine'

class SEARCH_TYPE(Enum):
    VECTOR = 'vector'
    HYBRID = 'hybrid'

class VSS(object):
    def __init__(self, args):
        self.connection: Connection = from_url(args.url)
        self.object_type: OBJECT_TYPE = args.objecttype
        self.index_type: INDEX_TYPE = args.indextype
        self.metric_type: METRIC_TYPE = args.metrictype
        self.image_dict: dict = {}
        self._vectorize()
        self._load_db()

    def _vectorize(self) -> None:
        """ Generates embeddings of images and writes them to file
        """    
        if (not os.path.exists(VECTOR_FILE) and len(os.listdir(IMAGE_DIR)) > 0):
            img2vec = Img2Vec(cuda=False)
            images: list = os.listdir(IMAGE_DIR)
            images = images[0:NUM_IMAGES]
            with open(VECTOR_FILE, 'w') as outfile:
                for image in images:
                    img: Image = Image.open(f'{IMAGE_DIR}/{image}').convert('RGB').resize((224, 224))
                    vector: list = img2vec.get_vec(img)
                    id: str = os.path.splitext(image)[0]
                    json.dump({'image_id': id, 'image_vector': vector.tolist()}, outfile)
                    outfile.write('\n')

    def _get_images(self) -> dict:
        """ Fetches image embeddings from a pre-made vector file

            Returns
            -------
            dictionary of image ids and their associated vectors
        """ 
        with open(VECTOR_FILE, 'r') as infile:
            for line in infile:
                obj: object = json.loads(line)
                id: str = str(obj['image_id'])
                match self.object_type:
                    case OBJECT_TYPE.HASH:
                        self.image_dict[id] = np.array(obj['image_vector'], dtype=np.float32).tobytes()
                    case OBJECT_TYPE.JSON:
                        self.image_dict[id] = obj['image_vector']  

    def _load_db(self) -> None:
        """ Loads Redis with hash sets containing image embeddings.  Creates an index containing a Vector field for the embedding
            and a user-specified text or tag field for the image id.
        """ 
        self.connection.flushdb()
        self._get_images()

        match self.object_type:
            case OBJECT_TYPE.HASH:
                schema = [ VectorField('image_vector', 
                                self.index_type.value, 
                                {   "TYPE": 'FLOAT32', 
                                    "DIM": 512, 
                                    "DISTANCE_METRIC": self.metric_type.value
                                }
                            ),
                            TagField('image_id')
                ]
                idx_def = IndexDefinition(index_type=IndexType.HASH, prefix=['key:'])
                self.connection.ft('idx').create_index(schema, definition=idx_def)

                pipe: Connection = self.connection.pipeline()
                for id, vec in self.image_dict.items():
                    pipe.hset(f'key:{id}', mapping={'image_id': id, 'image_vector': vec})
                pipe.execute()
            case OBJECT_TYPE.JSON:
                schema = [ VectorField('$.image_vector', 
                                self.index_type.value, 
                                {   "TYPE": 'FLOAT32', 
                                    "DIM": 512, 
                                    "DISTANCE_METRIC": self.metric_type.value
                                },  as_name='image_vector'
                            ),
                            TagField('$.image_id', as_name='image_id')
                ]
                idx_def: IndexDefinition = IndexDefinition(index_type=IndexType.JSON, prefix=['key:'])
                self.connection.ft('idx').create_index(schema, definition=idx_def)
                pipe: Connection = self.connection.pipeline()
                for id, vec in self.image_dict.items():
                    pipe.json().set(f'key:{id}', '$', {'image_id': id, 'image_vector': vec})
                pipe.execute()
    
    def randomImage(self) -> Tuple[str, list]:
        """ Generates a random image id and its associated vector

            Returns
            -------
            tuple containing a the id of the image and its associated vector
        """ 
        id, vector = random.choice(list(self.image_dict.items()))
        if self.object_type == OBJECT_TYPE.JSON:
            vector = np.array(vector, dtype=np.float32).tobytes()
        return id, vector
    
    def randomIDs(self, num: int) -> str:
        """ Generates a Redis Search query string.  String consists of logical ORs of
            image IDs.

            Parameters
            ----------
            num - number of IDs to fetch and join into a query string

            Returns
            -------
            query string
        """
        ids_to_query = random.sample(list(self.image_dict.keys()), num)
        ids_to_query = '|'.join(ids_to_query)
        return ids_to_query

    def search(self, query_vector: list, search_type: SEARCH_TYPE, hyb_str=None) -> list:
        """ Executes a straight vector or a hybrid search on Redis.

            Parameters
            ----------
            query_vector - vector for search
            search_type - vector or hybrid
            hyb_str - query string for a hybrid search

            Returns
            -------
            search result list
        """        
        match search_type:
            case SEARCH_TYPE.VECTOR:
                q_str = f'*=>[KNN {TOPK} @image_vector $vec_param AS vector_score]'
            case SEARCH_TYPE.HYBRID:
                q_str = f'(@image_id:{{{hyb_str}}})=>[KNN {TOPK} @image_vector $vec_param AS vector_score]'
        
        q = Query(q_str)\
            .sort_by('vector_score')\
            .paging(0,TOPK)\
            .return_fields('vector_score','image_id')\
            .dialect(2)    
        params_dict = {"vec_param": query_vector}

        results = self.connection.ft('idx').search(q, query_params=params_dict)
        return results

if __name__ == '__main__':
    parser = ArgumentParser(description='VSS Query Examples')
    parser.add_argument('--url', required=False, type=str, default=REDIS_URL,
        help='Redis URL connect string')
    parser.add_argument('--objecttype', required=False, 
        action=enum_action(OBJECT_TYPE), default=OBJECT_TYPE.JSON,
        help='Redis Object Type')
    parser.add_argument('--indextype', required=False,
        action=enum_action(INDEX_TYPE), default=INDEX_TYPE.FLAT, 
        help='Redis VSS Index Type')
    parser.add_argument('--metrictype', required=False,
        action=enum_action(METRIC_TYPE), default=METRIC_TYPE.L2,
        help='Redis VSS Metric Type')
    args = parser.parse_args()

    vss: VSS = VSS(args)
    id, vector = vss.randomImage()
    
    print('Vector Query')
    print(f'Query Vector Image ID:{id}\n')
    sres: list = vss.search(vector, SEARCH_TYPE.VECTOR)
    ids = []
    scores = []
    for doc in sres.docs:
        ids.append(doc.id.split(':')[1])
        scores.append(round(float(doc.vector_score),2))
    df = pd.DataFrame({'ID': ids, 'Score': scores})
    print(df.to_markdown(index=False))


    hyb_str: str = vss.randomIDs(5)
    print('\nHybrid Query')
    print(f'Query Vector Image ID:{id}')
    print(f'Hybrid Query String: {hyb_str}\n')
    sres = vss.search(vector, SEARCH_TYPE.HYBRID, hyb_str)
    ids = []
    scores = []
    for doc in sres.docs:
        ids.append(doc.id.split(':')[1])
        scores.append(round(float(doc.vector_score),2))
    df = pd.DataFrame({'ID': ids, 'Score': scores})
    print(df.to_markdown(index=False))