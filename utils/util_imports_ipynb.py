def converPyToIpynb(nbName):
    print('converPyToIpynb', nbName)
    !pip install ipython
    !pip install nbconvert
    #!ipython nbconvert — to script nbName '''converts to py'''
    print('installs done. Gonna convert')
    os.system(f'ipython nbconvert --to script {nbName}') 
    '''converts to py'''
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<DONE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

def importNotebook (nbName):
    print('importNotebook', nbName)
    if nbName.endswith('.py'):
        print('nbName ends with .py')
        fileExists = os.path.isfile(nbName)
        if fileExists:
            print('.py file exists')
            #import nbName
            open(nbName).read()
        else:
            print('.py files does not exist')
            nbName = nbName.replace("py", "ipynb")
            importNotebook(nbName)
    elif nbName.endswith('.ipynb'):
        print('nbName ends with .ipynb')
        fileExists = os.path.isfile(nbName)
        if fileExists:
            print('.ipynb file exists')
            converPyToIpynb(nbName)
            print('just converted ipynb to py')
            nbName = nbName.replace("ipynb", "py")
            print('nbName', nbName)
            importNotebook(nbName)
    else:
        print('nbName doesnt end with neither .py nor .ipynb')
        nbName = nbName + '.py'
        importNotebook(nbName)


def importNotebook2(nbName):
    import json
    if not nbName.endswith('.ipynb'):
        nbName = nbName + '.ipynb'
    
    with open(nbName) as json_file:
        myNB = json.load(json_file)
        nbCode = ''
        
        for nbCell in myNB['cells']:
            if nbCell['cell_type'] == 'code':
                for line in nbCell['source']:
                    nbCode = nbCode + line
                    #print('aqui', line)
                    #nbCode2 = compile(line, "<string>", "eval")

        return nbCode
