# DeepLearningModelsFastAPI
Serving Custome Deep learning model production-ready, fast, easy and secure powered by the great FastAPI

The code is still under development and the docker file will added soon.

To experiment and get a feeling on how to use this skeleton, a sample u-net model for market price prediction is included in this project. 
Follow the installation and setup instructions to run the sample model and serve it aso RESTful API.

<details><summary> # Requirements 
 
  > python 3.8+

<details><summary> # Setup
  
  > 1. Create a  .env file.
  > 2. In the .env file configure the API_KEY entry. The key is used for authenticating our API.
  
  
A sample API key can be generated using Python REPL:
```python
import uuid
print(str(uuid.uuid4()))
```
