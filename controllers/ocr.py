#elasticai_exec
import os
import numpy as np
import json
import sys

sys.path.insert(0, '..')
#where_am_i = os.system('hostname')
#hostname=os.popen('hostname').read()
#print('hostname', hostname)
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append("")

runApplication = True

if "RUNNING_LOCALLY" in os.environ.keys() and os.environ["RUNNING_LOCALLY"] == "True":
    print('RUNNING LOCALLY')
    #sys.path.insert(0, '..')
    #sys.path.append("")
    from utils.util_s3 import instantiateS3
    s3 = instantiateS3()

myDir = os.getcwd()
print('myDir', myDir)

parentDir = os.path.dirname(os.getcwd())
print('parentDir', parentDir)

os.chdir(parentDir) #mudar wd pra root
#sys.path.append("")


#s3DirOCR = os.path.join(parentDir, 'ocr')
#print('s3DirOCR', s3DirOCR)

if "RUNNING_LOCALLY" in os.environ.keys() and os.environ["RUNNING_LOCALLY"] == "True":
    print('A')
    #sys.path.append("")
    s3DirOCR = os.path.join(parentDir, 'ocr')
    platformDirOCR = os.path.join(os.getcwd(),'ocr', 'controllers')
 
else:
    print('B')
    #sys.path.append("")
    s3DirOCR = parentDir
    platformDirOCR = os.path.join(os.getcwd(), 'ocr')
    

#s3({"path": s3DirOCR, "localPath": '/'}, "readFolder")
s3({"path": parentDir, "localPath": '/'}, "readFolder")

print('files', os.listdir())


print('platformDirOCR', platformDirOCR)

print(' os.getcwd()',  os.getcwd())

os.chdir(platformDirOCR)
print(' os.getcwd()',  os.getcwd())
print('files', os.listdir())
#print('files', os.listdir())


#from object_detection import detect
print('checkpoint 1')
from utils.util_callFunction import callFunction
#print('checkpoint 2')
#import object_detection
#object_detection= __import__('object_detection')
#from char_recognition import recognize
print('checkpoint 3')


parent_directory = os.path.abspath('')
root = os.path.abspath(os.path.join(parent_directory, os.pardir))
data_output_folder = os.path.join(root, 'test', 'data', 'output')




#elasticai_exec
import datetime;
start = datetime.datetime.now()
print("current time:-", start)


objectsDetected = callFunction('object_detection.py', ['ex_001.png'], False, False)
print('objectsDetected', objectsDetected)

current = datetime.datetime.now()
print('diff time', current - start)







#elasticai_exec
#recognize()

chars_recognized = {"chars": []}

for imgIndex, imgResult in enumerate(objectsDetected['result']):
    print('imgIndex', imgIndex)
    #print('imgResult')
    for objectIndex, objectDetected in enumerate(imgResult['objBboxes']):
        #print(objectDetected)
        print('objectIndex', objectIndex)
        object_string = ''
        for charIndex, charDetected in enumerate(objectDetected['chars']):
            print('charIndex', charIndex)
            #print('charDetected', charDetected)
            #char_string = recognize(charDetected['snippet'])
            chars_recognized['chars'].append({})
            chars_recognized['chars'][charIndex]['index'] = charIndex
            #chars_recognized['chars'][charIndex]['char'] = callFunction('char_recognition.py', charDetected['snippet'], True)
            callFunction('char_recognition', charDetected['snippet'], True, True)
            #objectsDetected['result'][imgIndex]['objBboxes'][objectIndex]['chars'][charIndex]['text'] = chars_recognized['chars'][charIndex]['char']
            
            
            #object_string += chars_recognized['chars'][charIndex]['char']
            
            del objectsDetected['result'][imgIndex]['objBboxes'][objectIndex]['chars'][charIndex]['snippet']

        print('chars_recognized', chars_recognized)
        #object_string +=  ''.join([charDict['char'] for charDict in chars_recognized['chars'] if charDict.__contains__('char')])

        objectsDetected['result'][imgIndex]['objBboxes'][objectIndex]['text'] = object_string
        del objectsDetected['result'][imgIndex]['objBboxes'][objectIndex]['snippet']
    del objectsDetected['result'][imgIndex]['image']



    #elasticai_exec
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)





#elasticai_exec
#json_dump = json.dumps(objectsDetected, cls=NumpyEncoder)
outputLabelname = 'result.json'

with open(os.path.join(data_output_folder, outputLabelname), 'w') as fp:
    json.dump(objectsDetected, fp)




outputFilename = os.path.join('OCR', 'Final', 'test', 'data', 'output', outputLabelname)
outputFilename


s3({"path": outputFilename, "data": objectsDetected, "isFile": False }, "write")