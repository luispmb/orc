from typing import Optional
import sys

from fastapi import FastAPI, Request
import uvicorn
import os

app = FastAPI()


@app.get("/test")
def read_root():
    return {"Hello": "World"}


@app.post("/{controller}")
async def execute_controller(request: Request, controller):
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<HERE>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    #args = request.json().body
    args = await request.json()
    if args['args']:
        args = args['args']
    print('args', args)

    sys.path.insert(0, '..')
    sys.path.append("")

    print('cwd', os.getcwd())
    api_dir = os.path.join(os.getcwd(), 'controllers')
    print('api_dir', api_dir)
    os.chdir(api_dir) #mudar wd pra api_dir
    print('cwd', os.getcwd())

    from utils.util_callFunction import callFunction
    
    parentDir = os.path.dirname(os.getcwd())

    result = callFunction(controller, args, False, False)
    
    os.chdir(parentDir) #change back
    print('cwd', os.getcwd())

    return {"result": result}
    


@app.post("/*")
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
    return {"result": result}

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host="0.0.0.0")