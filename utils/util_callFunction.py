import os
import sys

# from grequests import asynchronous
#print('gonna import grequests')
#import grequests
import requests
from dotenv import load_dotenv
load_dotenv()

'''
where_am_i = os.getenv('WHEREAMI')
print('where_am_i', where_am_i)
where_am_i = os.system('hostname')
a=os.popen('hostname').read()
print(a.encode('utf-8'))
https://stackoverflow.com/questions/55339280/how-to-check-if-code-is-run-locally-or-on-a-cluster-in-python
'''



def callFunction(moduleFullName, args, direct, nonBlocking):
    print('moduleFullName', moduleFullName)

    where_am_i = os.system('hostname')
    hostname=os.popen('hostname').read()
    #print(hostname.encode('utf-8'))
    print('hostname', hostname)
    #print('type', type(hostname))

    if "RUNNING_LOCALLY" in os.environ.keys() and os.environ["RUNNING_LOCALLY"] == "True":
    #if not 'jobs' in hostname:
        # Do the local machine stuff
        print('RUNNING_LOCALLY')

        moduleName = moduleFullName
        if moduleFullName.endswith('.py') or moduleFullName.endswith('.ipynb'):
            moduleName = moduleFullName.split('.')[0]
        
        _module = __import__(moduleName)
        fn = getattr(_module, 'main')
        print('fn', fn)
        
        if nonBlocking == False:
            result = fn(args)
            return result
        else:
            # parallelization

            print('parallelization')
            #https://www.udacity.com/blog/2020/04/what-is-python-parallelization.html

            import threading
            # testar a paralelização para o recognize

            def numbers(start_num):
                for i in range(5):
                    print(start_num+i, end=' ')


            t1 = threading.Thread(target=numbers, args=(1,))
            print('args', args)
            t2 = threading.Thread(target=fn, args=(args,))
            #t1.start()
            t2.start()
            # wait for the processes to finish
            #t1.join()
            t2.join()
            # print a newline
            #print()


    else:
        # Do the ai platform stuff
        print('NOT RUNNING_LOCALLY')

        myDir = os.getcwd()
        print('myDir', myDir)

        parentDir = os.path.dirname(myDir)
        print('parentDir', parentDir)

        #controllersDir = os.path.join(parentDir, 'controllers')
        controllersDir = os.path.join(myDir, 'controllers')
        print('controllersDir', controllersDir)    

        moduleName = os.path.join(controllersDir, moduleFullName)
        print('moduleName', moduleName)


        # verificar se o ficheiro existe
        # se nao existir sacar todos os controllers e utils

        moduleExists = os.path.exists(moduleName)

        if not moduleExists:
            print('module', moduleName, ' is not here. Gonna download it')
           
            #download controllers and utils
            #dont need to instantiate s3 because we are in the platform which makes s3 default available
            s3({"path": controllersDir, "localPath": controllersDir, "isFile": False }, "read")
            utilsDir = os.path.join(parentDir, 'utils')
            print('utilsDir', utilsDir)
            s3({"path": utilsDir, "localPath": utilsDir, "isFile": False }, "read")



        moduleName = os.path.join(*moduleName.split('\\')[-3:])
        moduleName = moduleName.replace(os.sep, '/')
        print('moduleName', moduleName)
        #moduleName = os.path.join(parentFolder, 'controllers', )

        print('Gonna build request and invoke endpoint')
        data = buildJSONRequest(moduleName, args)
        endpoint = os.environ["API_ENDPOINT"]
        print('endpoint', endpoint)
        print('json data', data)
        
        if nonBlocking == True:
            response = requests.post(endpoint , json = data)
        else:
            
            # https://stackoverflow.com/questions/9110593/asynchronous-requests-with-python-requests
            print("a is greater than b")
            print('nonBlocking', nonBlocking)
            
            #from grequests import async
            response = requests.post(endpoint , json = data)
            
            '''
            urls = [endpoint]
            
            rs = (grequests.post(u, json = data) for u in urls)
            
            
            print('grequests.map(rs)', grequests.map(rs))
            '''
        #response = response.json()
        print('response - original', response)
        
        import json
        import numpy as np
        
        response = response.json()
        
        '''
        from flask import Flask, jsonify, make_response
        print('jsonify imported')
        
                
        app = Flask(__name__)
                
        with app.app_context():
            print('Gonna jsonificar')
            #response = jsonify(response)
            response = make_response(jsonify(response), 200)
        
        #response = jsonify(response)
        '''
        
        print('json response', response)

        
        output = response['output']
        print('output', output)
        
        return output


def buildJSONRequest(moduleName, args):  
    # data to be sent to api
    data = {
        "apikey": os.environ["PROJECT_API_KEY"],
        "image": os.environ["PROJECT_IMAGE"],
        "file": moduleName,
        "customOutput": "output",
        "queue": False,
        "polling": True,
        "args": args
    }

    return data