# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 19:14:27 2016

@author: puneet
"""

#conversion module from roman notations to SLP notations

def rom_slp(a):
        
    double_dict={}
    f=open('rom2.txt','r',encoding='utf8')

    #extracting words out of the lines of the text

    for lines in f.readlines():
        words=lines.split(',')
        words[1]=words[1].replace('\n','')
        double_dict[words[0]]=words[1]
    f.close()
    single_dict={}
    q=open('rom.txt','r',encoding='utf8')
    for lines in q.readlines():
        #print(lines)
        words=lines.split(',')
        words[1]=words[1].replace('\n','')
        single_dict[words[0]]=words[1]
    q.close()

    #change the value of 'a' through the following replacements.
    #for dictionaries single_dict and double_dict

    for elem in double_dict:
        if elem in a:
            a=a.replace(elem,double_dict[elem])
    for elem in single_dict:
        if elem in a:
            a=a.replace(elem,single_dict[elem])
    return(a)


