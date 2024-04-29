import sys, os, traceback
import ast
from statistics import mean

import torch
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
from sklearn.decomposition import PCA
import json
# from sklearn.manifold import TSNE
#Defining MAPE function
import csv

def fromDictToCsvColSelect(dictInput,csv_file,lstColumns):
    isOK=False
    try:
        dictOut=list(dictInput.values())

        # print('{} {}'.format(csv_columns,dictOut[0]))
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=lstColumns)
            writer.writeheader()
            for data in dictOut:
                writer.writerow(data)
        isOK=True
    except IOError:
        print("I/O error")
    return isOK
def fromDictToCsv(dictInput,csv_file):
    isOK=False
    try:
        dictOut=list(dictInput.values())
        csv_columns = list(dictOut[0].keys())

        # print('{} {}'.format(csv_columns,dictOut[0]))
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dictOut:
                writer.writerow(data)
        isOK=True
    except IOError:
        print("I/O error")
    return isOK

def revertOriginal(strLine,dictVocabs):
    lstWords=strLine.split()
    lstOutput=[]
    for word in lstWords:
        strId=word

        if word in dictVocabs.keys():
            strId=dictVocabs[word]
            # print('worrd {} {}'.format(word,strId))
            # input('sssss')
        lstOutput.append(strId)
    return ' '.join(lstOutput)

def preprocessLine(strLine):
    lstWords=strLine.split()
    lstOutput=[]
    for word in lstWords:
        wOut=word.replace('_L_','lhhs').replace('__','rhhs').replace('_','sphs')
        lstOutput.append(wOut)
    return ' '.join(lstOutput)

def convertToId(strLine,dictVocabs):
    lstWords=strLine.split()
    lstOutput=[]
    for word in lstWords:
        strId=''
        if not word in dictVocabs.keys():
            strNewId='w{}'.format(len(list(dictVocabs.keys()))+1)
            dictVocabs[word]=strNewId
        strId=dictVocabs[word]
        lstOutput.append(strId)
    return ' '.join(lstOutput)

def writeToFile(fp,arr,appendMode):
    f1=open(fp,appendMode)
    f1.write('\n'.join(arr)+'\n')
    f1.close()


def MAPE_cal(Y_actual,Y_Predicted):
    # mape = mean([abs((Y_actual[i] - Y_Predicted[i])/Y_actual[i]) for i in range(0,len(Y_actual))])
    mean_abs_percentage_error = MeanAbsolutePercentageError()
    mean_abs_percentage_error.cuda()
    y_act=torch.tensor(Y_actual).reshape(-1,1)
    y_act.cuda()
    y_pred=torch.tensor(Y_Predicted).reshape(-1,1)
    y_pred.cuda()
    mape=mean_abs_percentage_error(y_pred,y_act).item()
    # mape = torch.mean(abs((y_pred - y_act) / y_act)).cpu().item()
    # print('type mape {} {} {} {}'.format(y_act.dtype,y_pred.dtype,type(mape),mape))
    return mape

def fromFileToJSArray(file_path):
    data = []
    with open(file_path) as f:
        if 'jsonl' in file_path:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                if 'train' not in file_path:
                    if 'url' not in js.keys():
                        js['url'] = js['retrieval_idx']
                    if 'original_string' not in js.keys():
                        js['original_string'] = js['function']
                js['code']=js['original_string']
                data.append(js)
        elif "codebase" in file_path or "code_idx_map" in file_path:
            js = json.load(f)
            for key in js:
                temp = {}
                temp['original_string'] = key
                temp['code'] = key
                temp['code_tokens'] = key.split()
                temp["retrieval_idx"] = js[key]
                temp["url"] = js[key]
                temp['doc'] = ""
                temp['docstring_tokens'] = ""
                temp['keyToData'] = temp['url']
                data.append(temp)
        elif 'json' in file_path:
            for js in json.load(f):
                if 'url' not in js.keys():
                    js['url'] = js['retrieval_idx']
                if 'original_string' not in js.keys():
                    js['original_string'] = js['code']
                js['keyToData']=js['url']
                data.append(js)
    return data


def getXyFromDictWithSetFeaturesArrayReduction(indexSetFeature,dictInputFeatures,dictTargetValues,reductSize):
    keys=[]
    X=[]
    y=[]
    strExpSet='expS{}'.format(indexSetFeature)
    try:
        keyDictInputs=list(dictInputFeatures.keys())
        for keyI in keyDictInputs:
            if keyI in dictTargetValues.keys():
                vectorInput=dictInputFeatures[keyI][:reductSize]
                vectorTarget=dictTargetValues[keyI][strExpSet]
                # print('{} {}'.format(vectorInput,vectorTarget))
                # input('bbb')
                X.append(vectorInput)
                y.append(vectorTarget)
                # keys.append('{}__{}'.format(currentSplit,keyI))
                keys.append(keyI)

    except Exception as e:
        traceback.print_exc()
    return keys,X,y

def getXyFromDictWithSetFeaturesArray(indexSetFeature,dictInputFeatures,dictTargetValues):
    keys=[]
    X=[]
    y=[]
    strExpSet='expS{}'.format(indexSetFeature)
    try:
        keyDictInputs=list(dictInputFeatures.keys())
        for keyI in keyDictInputs:
            if keyI in dictTargetValues.keys():
                vectorInput=dictInputFeatures[keyI]
                vectorTarget=dictTargetValues[keyI][strExpSet]
                # print('{} {}'.format(vectorInput,vectorTarget))
                # input('bbb')
                X.append(vectorInput)
                y.append(vectorTarget)
                # keys.append('{}__{}'.format(currentSplit,keyI))
                keys.append(keyI)

    except Exception as e:
        traceback.print_exc()
    return keys,X,y


def getXyFromDictWithSetFeatures(indexSetFeature,dictInputFeatures,dictTargetValues):
    keys=[]
    X=[]
    y=[]
    strExpSet='expS{}'.format(indexSetFeature)
    try:
        keyDictInputs=list(dictInputFeatures.keys())
        for keyI in keyDictInputs:
            if keyI in dictTargetValues.keys():
                vectorInput=dictInputFeatures[keyI]
                vectorTarget=dictTargetValues[keyI][strExpSet]
                # print('{} {}'.format(vectorInput,vectorTarget))
                # input('bbb')
                X.append(vectorInput)
                y.append(vectorTarget)
                keys.append(keyI)

    except Exception as e:
        traceback.print_exc()
    Xarray=np.array(X)
    yarray=np.array(y)
    return keys,Xarray,yarray


def getXyFromDict(dictInputFeatures,dictTargetValues):
    keys=[]
    X=[]
    y=[]
    try:
        keyDictInputs=list(dictInputFeatures.keys())
        for keyI in keyDictInputs:
            if keyI in dictTargetValues.keys():
                vectorInput=dictInputFeatures[keyI]
                vectorTarget=dictTargetValues[keyI]['exp']
                # print('{} {}'.format(vectorInput,vectorTarget))
                # input('bbb')
                X.append(vectorInput)
                y.append(vectorTarget)
                keys.append(keyI)

    except Exception as e:
        traceback.print_exc()
    Xarray=np.array(X)
    yarray=np.array(y)
    return keys,Xarray,yarray


def getVectorFilteredByIndex(arrVector,setIndexes):
    lstOutputVector=[]
    for index in setIndexes:
        lstOutputVector.append(arrVector[index])
    return lstOutputVector

def exportDictToExcel(fpFile,lstRanks,dictInput):
    strHeader='Key,Rank,'
    lstTop1000=['Pos{}'.format(i) for i in range(1,1001)]
    strHeader=strHeader+','.join(lstTop1000)
    lstAllStrs=[strHeader]
    index=-1
    for key in dictInput.keys():
        index+=1
        val=dictInput[key]
        strLine='{},{},{}'.format(key,lstRanks[index],','.join(map(str,val)))
        lstAllStrs.append(strLine)
    f1=open(fpFile,'w')
    f1.write('\n'.join(lstAllStrs))
    f1.close()

def csm(A,B):
    num=np.dot(A,B.T)/(np.sqrt(np.sum(A**2,axis=1)[:,np.newaxis])*np.sqrt(np.sum(B**2,axis=1))[np.newaxis,:])
    return num


def extractAllTypePropertyForAST(jsonAST, arrCodeInfo):
    dictPerASTNonTerminalType={}
    dictPerASTTerminalType={}
    dictPerASTTerminalValue={}
    dictInformationPerDepth={}
    # for dep in lstDepthSelected:
    #     dictInformationPerDepth[dep]={}
    dictPerASTCountDepth={}
    # for idx in lstDepthSelected:
    #     dictInformationPerDepth[idx]={}
    #     dictPerASTCountDepth[idx]=0
    try:
        indent=0
        walkASTAndSaveDict(jsonAST, arrCodeInfo,indent,dictPerASTNonTerminalType,dictPerASTTerminalType,dictPerASTTerminalValue,dictInformationPerDepth,dictPerASTCountDepth)
    except Exception as e:
        dictSummarizeInfo=None
        traceback.print_exc()
    return dictPerASTNonTerminalType,dictPerASTTerminalType,dictPerASTTerminalValue,dictInformationPerDepth,dictPerASTCountDepth

def walkASTAndSaveDict(jsonAST, arrCodes,indent,dictPerASTNonTerminalType,dictPerASTTerminalType,dictPerASTTerminalValue,dictInformationPerDepth,dictPerASTCountDepth):
    if not isLeafNode(jsonAST):
        strTypeInfo=getPropReturnEmptyV2(jsonAST,'t')
        lstChildren=jsonAST['ci']

        if indent not in dictInformationPerDepth.keys():
            dictInformationPerDepth[indent]={}
            dictPerASTCountDepth[indent] = 0
        valIndent=dictInformationPerDepth[indent]
        # valCountPerDepth=dictPerASTCountDepth[indent]
        if strTypeInfo not in valIndent.keys():
            valIndent[strTypeInfo] = 1
        else:
            valIndent[strTypeInfo] += 1
        # if indent not in dictPerASTCountDepth.keys():
        #     dictPerASTCountDepth[indent]=0
        dictPerASTCountDepth[indent] += 1
        indentChild=indent+1
        if strTypeInfo not in dictPerASTNonTerminalType.keys():
            dictPerASTNonTerminalType[strTypeInfo]=1
        else:
            dictPerASTNonTerminalType[strTypeInfo] +=1
        for i in range(0,len(lstChildren)):
            child=lstChildren[i]
            walkASTAndSaveDict(child, arrCodes,indentChild,dictPerASTNonTerminalType,dictPerASTTerminalType,dictPerASTTerminalValue,dictInformationPerDepth,dictPerASTCountDepth)
    else:
        strTypeInfo = getPropReturnEmptyV2(jsonAST, 't')
        strValInfo=getTerminalValueFromASTNode(jsonAST,arrCodes)
        if strTypeInfo not in dictPerASTTerminalType.keys():
            dictPerASTTerminalType[strTypeInfo]=1
        else:
            dictPerASTTerminalType[strTypeInfo] +=1
        if strValInfo not in dictPerASTTerminalValue.keys():
            dictPerASTTerminalValue[strValInfo]=1
        else:
            dictPerASTTerminalValue[strValInfo] +=1
        if indent not in dictPerASTCountDepth.keys():
            dictPerASTCountDepth[indent]=0
        dictPerASTCountDepth[indent] += 1

def findMethodNode(astJson):
    dictFoundAST={}
    walkASTFindMethod(astJson,dictFoundAST)
    if 'ast' in dictFoundAST.keys():
        return dictFoundAST['ast']
    else:
        return astJson


def walkASTFindMethod(astJson,dictFoundAST):
    if not isLeafNode(astJson):

        lstChildren=astJson['ci']
        if astJson['t']=='method':
            dictFoundAST['ast']=astJson
        else:
            for child in lstChildren:
                walkASTFindMethod(child,dictFoundAST)

def findFunctionDeclarationNode(astJson):
    dictFoundAST={}
    walkASTFindFuncDeclr(astJson,dictFoundAST)
    if 'ast' in dictFoundAST.keys():
        return dictFoundAST['ast']
    else:
        return astJson


def walkASTFindFuncDeclr(astJson,dictFoundAST):
    if not isLeafNode(astJson):

        lstChildren=astJson['ci']
        if astJson['t']=='function_declaration':
            dictFoundAST['ast']=astJson
        else:
            for child in lstChildren:
                walkASTFindFuncDeclr(child,dictFoundAST)


def findFunctionDefinitionNode(astJson):
    dictFoundAST={}
    walkASTFindFuncDep(astJson,dictFoundAST)
    if 'ast' in dictFoundAST.keys():
        return dictFoundAST['ast']
    else:
        return astJson


def walkASTFindFuncDep(astJson,dictFoundAST):
    if not isLeafNode(astJson):

        lstChildren=astJson['ci']
        if astJson['t']=='function_definition':
            dictFoundAST['ast']=astJson
        else:
            for child in lstChildren:
                walkASTFindFuncDep(child,dictFoundAST)


def findMethodDeclarationNode(astJson):
    dictFoundAST={}
    walkASTFindMD(astJson,dictFoundAST)
    if 'ast' in dictFoundAST.keys():
        return dictFoundAST['ast']
    else:
        return astJson


def walkASTFindMD(astJson,dictFoundAST):
    if not isLeafNode(astJson):

        lstChildren=astJson['ci']
        if astJson['t']=='method_declaration':
            dictFoundAST['ast']=astJson
        else:
            for child in lstChildren:
                walkASTFindMD(child,dictFoundAST)

def getXyFromDictSubsetTargets(dictInputFeatures,dictTargetValues,lstSelectedIndexTargets):
    keys=[]
    X=[]
    y=[]
    try:
        keyDictInputs=list(dictInputFeatures.keys())
        for keyI in keyDictInputs:
            if keyI in dictTargetValues.keys():
                try:
                    vectorInput = dictInputFeatures[keyI]
                    vectorTarget = dictTargetValues[keyI]['exp']
                    vectorTarget=np.take(vectorTarget,lstSelectedIndexTargets)
                    # print('{} {}'.format(vectorInput,vectorTarget))
                    # input('bbb')
                    X.append(vectorInput)
                    y.append(vectorTarget)
                    keys.append(keyI)
                except Exception as e:
                    traceback.print_exc()


    except Exception as e:
        traceback.print_exc()
    Xarray=np.array(X)
    yarray=np.array(y)
    return keys,Xarray,yarray


def getXyFromDict(dictInputFeatures,dictTargetValues):
    keys=[]
    X=[]
    y=[]
    try:
        keyDictInputs=list(dictInputFeatures.keys())
        for keyI in keyDictInputs:
            if keyI in dictTargetValues.keys():
                vectorInput=dictInputFeatures[keyI]
                vectorTarget=dictTargetValues[keyI]['exp']
                # print('{} {}'.format(vectorInput,vectorTarget))
                # input('bbb')
                X.append(vectorInput)
                y.append(vectorTarget)
                keys.append(keyI)

    except Exception as e:
        traceback.print_exc()
    Xarray=np.array(X)
    yarray=np.array(y)
    return keys,Xarray,yarray

def adjustScoreForMatrix(scores):
    try:
        minValue=np.min(scores)
        maxValue=np.max(scores)
        distance=maxValue-minValue
        scores=(scores-minValue)/distance
    except Exception as e:
        traceback.print_exc()
    return scores


def getReductionEmb(nl_vecs,code_vecs,reductType,reductSize):
    lenNLVecs=len(nl_vecs)
    lenCodeVecs=len(code_vecs)
    nl_vecs_transform=[]
    code_vecs_transform=[]
    all_vecs=nl_vecs+code_vecs
    # print('len all {}'.format(len(all_vecs)))
    if reductType=='adhoc':
        all_vecs_transform=[element[:reductSize] for element in all_vecs]
        nl_vecs_transform = all_vecs_transform[:lenNLVecs]
        code_vecs_transform = all_vecs_transform[lenNLVecs:]
    elif reductType=='pca':
        pca = PCA(n_components=reductSize)
        all_vecs_transform =pca.fit_transform(all_vecs)
        nl_vecs_transform = all_vecs_transform[:lenNLVecs]
        code_vecs_transform = all_vecs_transform[lenNLVecs:]
    # else:
    #     tsne = TSNE(n_components=reductSize,
    #                 perplexity=40,
    #                 random_state=42,
    #                 n_iter=5000,
    #                 n_jobs=-1)
    #     all_vecs_transform = tsne.fit_transform(all_vecs)
    #     nl_vecs_transform = all_vecs_transform[:lenNLVecs]
    #     code_vecs_transform = all_vecs_transform[lenNLVecs:]

    return nl_vecs_transform.tolist(),code_vecs_transform.tolist()


def getMAEScoreDictPred(dictPred):
    lstKeys=[]
    lstExpecteds=[]
    lstPredicteds=[]
    for key in dictPred.keys():
        vectorExp=dictPred[key]['exp']
        vectorPred=dictPred[key]['pred']
        lstKeys.append(key)
        lstExpecteds.append(vectorExp)
        lstPredicteds.append(vectorPred)
    lstTotalMAEs=[]
    numFeatures=len(lstExpecteds[0])
    # print('num ent {}\n{}'.format(lstPredicteds,lstExpecteds))
    for indexFeature in range(0,numFeatures):
        lstExpFeatIndex=[item[indexFeature] for item in lstExpecteds]
        lstPredFeatIndex = [item[indexFeature] for item in lstPredicteds]
        maeFeat=mean_absolute_error(lstExpFeatIndex,lstPredFeatIndex)
        lstTotalMAEs.append(maeFeat)
    avgMAE=mean(lstTotalMAEs)
    lstTotalMAEs.append(avgMAE)
    return lstTotalMAEs

def getMAEAndMSEScoreDictPred(dictPred):
    lstKeys=[]
    lstExpecteds=[]
    lstPredicteds=[]
    for key in dictPred.keys():
        vectorExp=dictPred[key]['exp']
        vectorPred=dictPred[key]['pred']
        lstKeys.append(key)
        lstExpecteds.append(vectorExp)
        lstPredicteds.append(vectorPred)
    lstTotalMAEs=[]
    lstTotalMSEs = []
    numFeatures=len(lstExpecteds[0])
    # print('num ent {}\n{}'.format(lstPredicteds,lstExpecteds))
    for indexFeature in range(0,numFeatures):
        lstExpFeatIndex=[item[indexFeature] for item in lstExpecteds]
        lstPredFeatIndex = [item[indexFeature] for item in lstPredicteds]
        maeFeat=mean_absolute_error(lstExpFeatIndex,lstPredFeatIndex)
        mseFeat = mean_squared_error(lstExpFeatIndex, lstPredFeatIndex)
        lstTotalMAEs.append(maeFeat)
        lstTotalMSEs.append(mseFeat)
    avgMAE=mean(lstTotalMAEs)
    avgMSE=mean(lstTotalMSEs)
    lstTotalMAEs.append(avgMAE)
    lstTotalMSEs.append(avgMSE)
    return lstTotalMAEs,lstTotalMSEs

def getMAEAndRMSEScoreDictPred(dictPred):
    lstKeys=[]
    lstExpecteds=[]
    lstPredicteds=[]
    for key in dictPred.keys():
        vectorExp=dictPred[key]['exp']
        vectorPred=dictPred[key]['pred']
        lstKeys.append(key)
        lstExpecteds.append(vectorExp)
        lstPredicteds.append(vectorPred)
    lstTotalMAEs=[]
    lstTotalRMSEs = []
    numFeatures=len(lstExpecteds[0])
    # print('num ent {}\n{}'.format(lstPredicteds,lstExpecteds))
    for indexFeature in range(0,numFeatures):
        lstExpFeatIndex=[item[indexFeature] for item in lstExpecteds]
        lstPredFeatIndex = [item[indexFeature] for item in lstPredicteds]
        maeFeat=mean_absolute_error(lstExpFeatIndex,lstPredFeatIndex)
        rmseFeat = pow(mean_squared_error(lstExpFeatIndex, lstPredFeatIndex),0.5)
        lstTotalMAEs.append(maeFeat)
        lstTotalRMSEs.append(rmseFeat)
    # avgMAE=mean(lstTotalMAEs)
    # avgRMSE=mean(lstTotalRMSEs)
    # lstTotalMAEs.append(avgMAE)
    # lstTotalRMSEs.append(avgRMSE)
    return lstTotalMAEs,lstTotalRMSEs

def getMAEScore(dictExp,dictPred):
    lstKeys=[]
    lstExpecteds=[]
    lstPredicteds=[]
    for key in dictExp.keys():
        if key in dictPred.keys():
            vectorExp=dictExp[key]['exp']
            vectorPred=dictPred[key]['pred']
            lstKeys.append(key)
            lstExpecteds.append(vectorExp)
            lstPredicteds.append(vectorPred)
    lstTotalMAEs=[]
    numFeatures=len(lstExpecteds[0])
    for indexFeature in range(0,numFeatures):
        lstExpFeatIndex=[item[indexFeature] for item in lstExpecteds]
        lstPredFeatIndex = [item[indexFeature] for item in lstPredicteds]
        maeFeat=mean_absolute_error(lstExpFeatIndex,lstPredFeatIndex)
        lstTotalMAEs.append(maeFeat)
    avgMAE=mean(lstTotalMAEs)
    lstTotalMAEs.append(avgMAE)
    return lstTotalMAEs





strJavaHeaderStart='class HelloWorld {'
strJavaHeaderEnd='}'
def  getTemplateClassForJavaCode(strRawCode):
    strFinalCode=strRawCode
    try:
        arrCodes=strRawCode.split('\n')
        lstOutCode=[strJavaHeaderStart]
        for code in arrCodes:
            strNewLine='\t{}'.format(code)
            lstOutCode.append(strNewLine)
        lstOutCode.append(strJavaHeaderEnd)
        strFinalCode='\n'.join(lstOutCode)
    except Exception as e:
        traceback.print_exc()
    return strFinalCode

def  getTemplateClassForPHPCode(strRawCode):
    strFinalCode=strRawCode
    try:
        arrCodes=strRawCode.split('\n')
        # lstOutCode=['<html>\n<body>\n\n<?php']
        lstOutCode = ['<?php']
        for code in arrCodes:
            strNewLine='\t{}'.format(code)
            lstOutCode.append(strNewLine)
        # lstOutCode.append('?>\n\n</body>\n</html>')
        lstOutCode.append('?>')
        strFinalCode='\n'.join(lstOutCode)
    except Exception as e:
        traceback.print_exc()
    return strFinalCode


def getASTAndShowException(strCode,currentLanguageParser):
    dictJsonAST={}
    strItemAST=''
    isOK=False
    try:
        # currentLanguageParser = dictConfig['currentLanguageParser']
        tree = currentLanguageParser.parse(bytes(strCode, 'utf8'))
        cursor = tree.walk()
        node = cursor.node
        listId=[]
        dictJsonAST = walkTreeAndReturnJSonObject(node, strCode.split('\n'), listId)
        strItemAST=str(dictJsonAST)
        isOK=True
    except Exception as e:
        traceback.print_exc()
    return dictJsonAST,strItemAST,isOK

def walkTreeAndReturnJSonObject(node,arrCodes,listId):
    dictJson={}
    strType=str(node.type)
    # print(strType)
    dictJson['t']=strType
    dictJson['id'] = len(listId)+1
    listId.append(len(listId)+1)
    strStart=str(node.start_point)
    strEnd = str(node.end_point)
    arrStart = strStart.split(',')
    arrEnd = strEnd.split(',')
    startLine = int(arrStart[0].replace('(', ''))
    startOffset = int(arrStart[1].replace(')', ''))
    endLine = int(arrEnd[0].replace('(', ''))
    endOffset = int(arrEnd[1].replace(')', ''))
    dictJson['sl']=startLine
    dictJson['so'] = startOffset
    dictJson['el'] = endLine
    dictJson['eo'] = endOffset

    # if strType!='translation_unit' and endLine<33:
    #     return dictJson
    listChildren=node.children

    if listChildren is not None and len(listChildren)>0:
        dictJson['ci'] = []
        for i in range(0,len(listChildren)):

            arrChildEnd = str(listChildren[i].end_point).split(',')
            endChildLine = int(arrChildEnd[0].replace('(', ''))
            # if endChildLine>=33:
            childNode = walkTreeAndReturnJSonObject(listChildren[i], arrCodes, listId)
            dictJson['ci'].append(childNode)
    return dictJson

def writeDictValueToFile(dictInput,fpFile):
    listKeys=list(dictInput.keys())
    lstVals=['{}\t{}'.format(key,dictInput[key]) for key in listKeys]
    f1=open(fpFile,'w')
    f1.write('\n'.join(lstVals))
    f1.close()

def createDirIfNotExist(fopOutput):
    try:
        # Create target Directory
        os.makedirs(fopOutput, exist_ok=True)
        #print("Directory ", fopOutput, " Created ")
    except FileExistsError:
        print("Directory ", fopOutput, " already exists")

def getPropReturnEmpty(jsonObject,strType):
    strOutput='{}:""'.format(strType)
    if strType in jsonObject.keys():
        strTypeDisplay=strType
        if strType=='ta':
            strTypeDisplay='type'
        elif strType=='t':
            strTypeDisplay='type'
        elif strType=='val':
            strTypeDisplay='value'
        strOutput='{} : {}'.format(strTypeDisplay,jsonObject[strType])
    return strOutput
def getPropReturnEmptyV2(jsonObject,strType):
    strOutput=''
    if strType in jsonObject.keys():
        strOutput='{}'.format(jsonObject[strType])
    return strOutput

def isLeafNode(jsonObject):
    isLeaf=True
    if 'ci' in jsonObject.keys() and len(jsonObject['ci'])>0:
        isLeaf=False
    return isLeaf




def extractPropertyForAST(strJsonAST, arrCodeInfo):
    dictNodeTypesPerCandidate={}
    dictValuesPerCandidate = {}
    lstLeafNodesAndParentalNodes = []
    dictSummarizeInfo={}
    dictSummarizeInfo['NumOfNonterminalNodes']=0
    dictSummarizeInfo['NumOfTerminalNodes'] = 0
    dictSummarizeInfo['DepthOfAST']=1
    dictSummarizeInfo['AverageOfDepthOfPaths'] = 1
    dictSummarizeInfo['AverageOfChildrenPerNonTerminalNodes'] = 1
    dictSummarizeInfo['ListOfNumOfChildrenInBranch'] = []

    try:
        indent=-1
        jsonAST=ast.literal_eval(strJsonAST)
        # dictNodeTypesPerCandidate={}
        lstAncestorPaths=[]
        dictIdCurrentNodeNumbers={}
        dictIdCurrentNodeNumbers['id']=0
        walkASTAndEstimate(jsonAST, arrCodeInfo,indent,dictIdCurrentNodeNumbers,lstAncestorPaths,lstLeafNodesAndParentalNodes, dictNodeTypesPerCandidate,dictValuesPerCandidate,dictSummarizeInfo)
        dictSummarizeInfo['AverageOfChildrenPerNonTerminalNodes'] = mean(dictSummarizeInfo['ListOfNumOfChildrenInBranch'])
        lstPathsOutput=[len(lstItem) for lstItem in lstLeafNodesAndParentalNodes]
        dictSummarizeInfo['DepthOfAST'] = max(lstPathsOutput)-1
        dictSummarizeInfo['AverageOfDepthOfPaths'] = mean(lstPathsOutput) - 1
        dictSummarizeInfo.pop('ListOfNumOfChildrenInBranch')
    except Exception as e:
        dictSummarizeInfo.pop('ListOfNumOfChildrenInBranch')
        dictSummarizeInfo=None
        traceback.print_exc()
    return dictNodeTypesPerCandidate,dictValuesPerCandidate,lstLeafNodesAndParentalNodes,dictSummarizeInfo

def walkASTAndEstimate(jsonAST, arrCodes,indent,dictIdCurrentNodeNumbers,lstAncestorPaths,lstLeafNodesAndParentalNodes, dictNodeTypesPerCandidate, dictValuesPerCandidate,dictSummarizeInfo):
    if not isLeafNode(jsonAST):
        strTypeInfo=getPropReturnEmptyV2(jsonAST,'t')
        if strTypeInfo!='':
            if strTypeInfo not in dictNodeTypesPerCandidate.keys():
                dictNodeTypesPerCandidate[strTypeInfo]=0
            dictNodeTypesPerCandidate[strTypeInfo]+=1
            # if strTypeInfo not in dictStatASTAll.keys():
            #     dictStatASTAll[strTypeInfo]=0
            # dictStatASTAll[strTypeInfo]+=1

        dictCurrentNodeInfo = {}
        dictCurrentNodeInfo['id'] = dictIdCurrentNodeNumbers['id']
        dictCurrentNodeInfo['ide'] = indent
        dictCurrentNodeInfo['t'] = strTypeInfo
        lstAncestorPaths.insert(0, dictCurrentNodeInfo)

        lstChildren=jsonAST['ci']
        dictSummarizeInfo['ListOfNumOfChildrenInBranch'].append(len(lstChildren))
        dictSummarizeInfo['NumOfNonterminalNodes'] += 1

        for i in range(0,len(lstChildren)):
            child=lstChildren[i]
            walkASTAndEstimate(child, arrCodes,indent,dictIdCurrentNodeNumbers,lstAncestorPaths,lstLeafNodesAndParentalNodes, dictNodeTypesPerCandidate,dictValuesPerCandidate,dictSummarizeInfo)
        del lstAncestorPaths[0]
    else:
        strTypeInfo = getPropReturnEmptyV2(jsonAST, 't')
        strValInfo=getTerminalValueFromASTNode(jsonAST,arrCodes)

        lstCopyPaths = lstAncestorPaths.copy()
        dictCurrentNodeInfo = {}
        dictCurrentNodeInfo['id'] = dictIdCurrentNodeNumbers['id']
        dictCurrentNodeInfo['ide'] = indent
        dictCurrentNodeInfo['t'] = strTypeInfo
        dictCurrentNodeInfo['val'] = strValInfo
        lstCopyPaths.insert(0, dictCurrentNodeInfo)
        lstLeafNodesAndParentalNodes.append(lstCopyPaths)

        if strTypeInfo not in dictNodeTypesPerCandidate.keys():
            dictNodeTypesPerCandidate[strTypeInfo] = 0
        dictNodeTypesPerCandidate[strTypeInfo] += 1
        # if strTypeInfo not in dictStatASTAll.keys():
        #     dictStatASTAll[strTypeInfo] = 0
        # dictStatASTAll[strTypeInfo] += 1

        # if strValInfo not in dictStatTerminalValuesAll.keys():
        #     dictStatTerminalValuesAll[strValInfo]=0
        # dictStatTerminalValuesAll[strValInfo]+=1
        if strValInfo not in dictValuesPerCandidate.keys():
            dictValuesPerCandidate[strValInfo]=0
        dictValuesPerCandidate[strValInfo] +=1
        dictSummarizeInfo['NumOfTerminalNodes'] += 1
        # dictNumTokens['Num']+=1


def analyzeContentDict(strJsonAST, arrCodeInfo, dictStatNodeTypesAll, dictStatValuesAll):
    dictNodeTypesPerCandidate={}
    dictValuesPerCandidate = {}
    try:
        indent=-1
        jsonAST=ast.literal_eval(strJsonAST)
        dictNodeTypesPerCandidate={}
        # dictNumTokens={}
        # dictNumTokens['Num']=0
        walkASTAndStat(jsonAST, arrCodeInfo,  dictStatNodeTypesAll,dictStatValuesAll, dictNodeTypesPerCandidate,dictValuesPerCandidate)

    except Exception as e:
        traceback.print_exc()
    return dictNodeTypesPerCandidate,dictValuesPerCandidate

def walkASTAndStat(jsonAST, arrCodes, dictStatASTAll, dictStatTerminalValuesAll, dictNodeTypesPerCandidate, dictValuesPerCandidate):
    if not isLeafNode(jsonAST):
        strTypeInfo=getPropReturnEmptyV2(jsonAST,'t')
        if strTypeInfo!='':
            if strTypeInfo not in dictNodeTypesPerCandidate.keys():
                dictNodeTypesPerCandidate[strTypeInfo]=0
            dictNodeTypesPerCandidate[strTypeInfo]+=1
            if strTypeInfo not in dictStatASTAll.keys():
                dictStatASTAll[strTypeInfo]=0
            dictStatASTAll[strTypeInfo]+=1
        lstChildren=jsonAST['ci']
        for i in range(0,len(lstChildren)):
            child=lstChildren[i]
            walkASTAndStat(child, arrCodes, dictStatASTAll,dictStatTerminalValuesAll, dictNodeTypesPerCandidate,dictValuesPerCandidate)
    else:
        strTypeInfo = getPropReturnEmptyV2(jsonAST, 't')
        strValInfo=getTerminalValueFromASTNode(jsonAST,arrCodes)

        if strTypeInfo not in dictNodeTypesPerCandidate.keys():
            dictNodeTypesPerCandidate[strTypeInfo] = 0
        dictNodeTypesPerCandidate[strTypeInfo] += 1
        if strTypeInfo not in dictStatASTAll.keys():
            dictStatASTAll[strTypeInfo] = 0
        dictStatASTAll[strTypeInfo] += 1

        if strValInfo not in dictStatTerminalValuesAll.keys():
            dictStatTerminalValuesAll[strValInfo]=0
        dictStatTerminalValuesAll[strValInfo]+=1
        if strValInfo not in dictValuesPerCandidate.keys():
            dictValuesPerCandidate[strValInfo]=0
        dictValuesPerCandidate[strValInfo] +=1
        # dictNumTokens['Num']+=1

def getTerminalValueFromASTNode(jsonInput,arrCodes):
    strReturn=''
    try:
        lstStr = []
        startPointLine = jsonInput['sl']
        startPointOffset = jsonInput['so']
        endPointLine = jsonInput['el']
        endPointOffset = jsonInput['eo']

        if startPointLine == endPointLine:
            return arrCodes[startPointLine][startPointOffset:endPointOffset]
        for i in range(startPointLine, endPointLine + 1):

            if i == startPointLine:
                strAdd = arrCodes[i][startPointOffset:]
                lstStr.append(strAdd)
            elif i == endPointLine:
                strAdd = arrCodes[i][:endPointOffset]
                lstStr.append(strAdd)
            else:
                strAdd = arrCodes[i]
                lstStr.append(strAdd)
        strReturn = '\n'.join(lstStr).strip()
    except Exception as e:
        pass
    return strReturn
