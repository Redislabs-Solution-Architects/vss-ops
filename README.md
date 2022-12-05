# VSS Examples
Examples of Vector Similarity Search (VSS) in Python 
## Features
- Provides examples of how to conduct straight vector and hybrid VSS searches.
## Prerequisites
- Python
- Download of images from Kaggle site - https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
- RediSearch 2.6/Redis JSON 2.4.  Dockerfile included will build a workable environment.
fashion-product-images-dataset
## Installation
1. Clone this repo.

2. Create an images folder in the data folder and download the images from the URL above.

3. Install Python requirements (either in a virtual env or global)
```bash
pip install -r requirements.txt
```
## Usage
### Options
- --url. Redis connection string.  Default = redis://localhost:6379
- --objecttype. Redis Object Type.  Default = json
- --indextype. Redis VSS Index Type. Default = flat
- --metrictype. Redis VSS Metric Type.  Default = l2
### Execution
```bash
python3 example.py --objecttype json --indextype hnsw --metrictype cosine
```
### Output

Vector Query
Query Vector Image ID:58452

|    ID |   Score |
|------:|--------:|
| 58452 |    0    |
| 31863 |    0.12 |
| 46314 |    0.13 |
| 33909 |    0.22 |
| 22945 |    0.24 |

Hybrid Query
Query Vector Image ID:58452

Hybrid Query String: 2107|2656|25155|40953|34242

|    ID |   Score |
|------:|--------:|
|  2656 |    0.4  |
| 40953 |    0.42 |
| 34242 |    0.49 |
| 25155 |    0.51 |
|  2107 |    0.52 |
