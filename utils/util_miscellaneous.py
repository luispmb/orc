#!/usr/bin/env python
# coding: utf-8

def changeDirectory(destiny_folder):
    cwd = os.getcwd()
    csd = os.path.join(cwd, destiny_folder)
    print(csd)
    #csd = 'C:\\Users\\Administrator\\OCR\\Final\\01-Data'
    os.chdir(csd)
    print(os.getcwd())
    #counter+=1