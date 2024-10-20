import os
import numpy as np
import sys
import re
from collections import defaultdict
if __name__ == "__main__":
    path=r'C:\Users\mm200507d\Desktop\psiml\pythonProject\public\public\set\9'
    #path=input()
    files=os.listdir(path)
    log_files=[]
    stack=[]
    for file in files:
        stack.append(os.path.join(path,file))
    cnt=0
    while(len(stack)>0):
        curr=stack.pop()
        #print(curr)
        if os.path.isdir(curr):
            files=os.listdir(curr)
            for file in files:
                stack.append(os.path.join(curr,file))
        else:
            if curr.endswith('logtxt'):
                cnt+=1
                log_files.append(curr)

    print(cnt)

    entries=0
    for log in log_files:
        with open(log, 'r') as file:
            for line in file:
                if not (line.isspace()):
                    entries+=1


    error_entries=0
    for log in log_files:
        flag=0
        with open(log, 'r') as file:
            for line in file:
                if '[error]' in line or 'loglevel=error' in line or \
                '<err>' in line or 'level=ERROR' in line or 'fatal-error' in line:
                    error_entries+=1
                    flag=1
                    break
            if (flag==1):
                continue


    print(entries)
    print(error_entries)


    #sent1="2024 02 24 19:54:34 QuantumComputeServices: <err> File not found on server"
    #sent2="24.02.2024.19h:56m:12s information StratoVirtualHost --- Password reset request submitted"

    #reg1 = r'\d{4} \d{2} \d{2} \d{2}:\d{2}:\d{2} \b\w+\b: \<\b\w+\b\> (.*)'
    #reg2 = r'\d{2}\.\d{2}\.\d{4}\.\d{2}h:\d{2}m:\d{2}s \w+ \w+ --- (.*)'

    #sent3 = "dt=2024-02-24_20:06:06 level=DEBUG service=VaultEncryptionService msg=Allocated memory for processing"
    #sent4="24.02.2024.20:10:40 CEF:0|StratoVirtualHost|loglevel=info msg=New API key generated"
    #sent5="[2024-02-24 20:18:55] [error] [HorizonCloudServices] - Email delivery failure"

    #matches=re.search(reg4,sent5)

    #print(matches.group(1))
    # mo2 = re.match('(011)([0-9]*)', telefon)

    reg1=r'\> (.*)$'
    reg2=r'--- (.*)$'
    reg3=r'msg=(.*)$'
    reg4=r' - (.*)$'

    reg_list=[reg1,reg2,reg3,reg4]

    word_dict={}

    for i,log in enumerate(log_files,0):
        print("----------------------------------------")
        print(log)

        with open(log, 'r') as file:
            for line in file:
                if not (line.isspace()):
                    for reg in reg_list:
                        matches = re.search(reg, line)
                        if matches:
                            body=matches.group(1)
                            body = re.sub(r'[^\w\s]', '', body)
                            print(body)
                            for key in word_dict:
                                word_dict[key][1]=0
                            for word in body.split():
                                if word not in word_dict:
                                    word_dict[word] = [1,1]

                                else:
                                    if (word_dict[word][1]==0):
                                        word_dict[word][0]+=1
                                        word_dict[word][1]=1
                            break

    sorted_dict=sorted(word_dict.items(), key=lambda item: (-item[1][0],item[0]))


    #print(word_dict)
    print(sorted_dict)

    cnt2=0

    for item in sorted_dict:
        if(cnt2>=5): break
        if cnt2!=4:
            print(f"{item[0]}, ",end="")
        else:
            print(f"{item[0]}")
        cnt2+=1










