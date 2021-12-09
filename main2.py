from typing import Optional

from fastapi import FastAPI
import uvicorn
import os

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/api/{controller}")
def read_item(controller: str, q: Optional[str] = None):
    moduleName = controller
    if controller.endswith('.py') or controller.endswith('.ipynb'):
        moduleName = controller.split('.')[0]
    
    controllersDir = os.path.join(os.getcwd(), 'controllers')
    os.chdir(controllersDir) #mudar wd pra controllersDir
    
    _module = __import__(moduleName)
    fn = getattr(_module, 'main')
    print('fn', fn)
    result = fn(args) 
    return {"result": result}a

if __name__ == '__main__':
    uvicorn.run(app)