import os
import sys
from dotenv import load_dotenv
load_dotenv()

def instantiateS3():
    print('instantiateS3')
    try:
        #if not 's3' in locals() or not 's3' in globals():
            #print('s3 not instantiated yet')
        global s3
        from elasticai_s3 import storage_s3 as s3        
        os.environ["BUCKET"] = os.environ.get("S3_BUCKET")
        os.environ["API_KEY"] = os.getenv('S3_API_KEY')
        os.environ["SERVICE_INSTANCE"] = os.getenv('S3_SERVICE_INSTANCE')
        os.environ["AUTH_ENDPOINT"] = os.getenv('S3_AUTH_ENDPOINT')
        os.environ["ENDPOINT"] = os.getenv('S3_ENDPOINT')
        print('s3 instantiated')

    except:
        print('gonna install elasticai_s3 s3')
        import pip
        pip.main(['install'], "git+https://eu-de.git.cloud.ibm.com/miguel.vasques/elasticai-s3.git")
        #from elasticai_s3 import storage_s3 as s3
        instantiateS3()

    return s3