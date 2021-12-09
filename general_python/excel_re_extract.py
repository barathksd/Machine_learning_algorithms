# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:42:56 2020

@author: 81807
"""
import pandas as pd
import numpy as np
import pandas as pd
from pathlib import Path
import re
import json
import os
from collections import Counter
from datetime import datetime
import datetime as dtlib

# ファイルの読み込み
basepath = 'C:\\Users\\81807\\Documents\\DWIBS\\'
def load_data(file = basepath+'CAFI_Report.xlsx'):
    df = pd.read_excel(file)
    
    # 病院番号・名前の統合
    dic_host = {601:'広島',602:'北福島',603:'有吉',604:'えだ',605:'おさか',\
                606:'みたか',607:'焼津',608:'八王子',609:'都島',610:'鎌倉',\
                611:'東邦',612:'道東',613:'岡山',614:'大久保',615:'白十字',\
                616:'銚子', 617:'東京北部',618:'大手町',619:'放射線P',620:'temp',621:'三上HP'}
        
    #病院リストを作成
    list_host = [dic_host[int(str(pat_id)[:3])] for pat_id in df['患者ID']]
    df['病院'] = pd.DataFrame(list_host)
    df['colid'] = np.arange(len(df),0,-1)
    
    return df

# column IDが一致する行を出力する関数
getrow = lambda df,colid: df[df['colid']==colid]

df = load_data()   # pandas dataframe 形式のデータ

#  ---   診断の分析   --- 
def create1(df):
    #s = ''
    
    d = {}
    for index,row in enumerate(df['診断']):
        d[index] = {}
        row = row.replace('　',' ').replace('：',':')   # 全角文字を半角にする
        mark = row.find('※')
        if mark>0:
            d[index]['注意'] = row[mark:mark+row[mark:].find('\n')]
            row = row.replace(d[index]['注意'],'')
        i = row.find('参考')
        if i>1:
            row = row[:i-2]
        
        # パターンの定義、位置する分をもらって保存する
        match = re.match('.*判定 *(?P<判定>\w):\n* *(?P<判定詳細>\w+((\(|（)\w+(\)|\）))?\w*。?) *(?P<位置>(\(|（).+(\)|\）).?)?\n+(?P<内容>.*)',row,flags=re.DOTALL)
        if match == None:
            print(index,mark,row)
            #s = row
            continue
        
        d[index]['colid'] = df.iloc[index]['colid']
        d[index]['判定'] = match['判定']
        d[index]['判定詳細'] = match['判定詳細']
        d[index]['位置'] = match['位置']
        d[index]['内容'] = match['内容']
    
    return d

#  ---    所見の分析    ---  

#s = df['所見'].iloc[234]
def create2(df):
    p = re.compile(r'(\(|\（)図[1-9](\)|\）)')
    d = {}
    
    for index,s in enumerate(df['所見']):
        
        d[index] = {}
        m = 0
        for s2 in s.split('\n\n'):
            
            match = re.match(' *\n*(【\w+】.*)',s2)
            if match:
                d[index]['方法'] = match.group()
                match = None
            
            elif sum(1 for _ in p.finditer(s2)) == 0:
                d[index]['misc'+str(m)] = s2
                m += 1
                continue
            
            l = []
            k = []
            
            for match in p.finditer(s2):
                 start,end = (int(match.start()),int(match.end()))
                 l.append((start,end))
                 k.append(s2[start:end])
                 
            next0 = 0
            for i in range(len(l)):
                if i<len(l)-1:
                    finddot = s2[l[i][1]:l[i+1][0]].rfind('。')
                    #print(i,l[i],finddot,s2[l[i][1]:l[i+1][0]],k[i])
                    if finddot <0 and i < len(l) -2:
                        finddot = s2[l[i][1]:l[i+2][0]].rfind('。')
                        end1 = l[i][1]+finddot
                        k[i] = k[i]+k[i+1]
                        d[index][k[i]] = s2[next0:end1]+'。'
                        i += 1
                        continue
                        
                    elif finddot>=0:
                        end1 = l[i][1]+finddot
                        
                    else:
                        k[i] = k[i]+k[i+1]
                        end1 = len(s2)
                        d[index][k[i]] = s2[next0:end1]
                        break
                        
                elif i == len(l)-1:
                    end1 = len(s2)
                d[index][k[i]] = s2[next0:end1]
                next0 = end1 + 1
    
    return d

#  ---    検査目的    ------

#　項目の定義
template = {'依頼医':0,
            '技師名':1,'担当技師':1,'撮影技師':1,
            '受診理由':2,'理由':2,
            '検診有無':3,'乳がん健診歴':3,'乳がん検診歴':3,'乳がん検診の受診歴':3,'検査歴':3,'乳がん検診受診歴の有無':3,'乳癌検診歴':3,
            '既往歴':4,'乳房・卵巣既往歴':4,'治療歴':4,'乳房に関する既往歴':4,'乳房・卵巣疾患':4,'乳房・婦人科疾患に関する既往歴':4,
            '授乳歴':5,'授乳経験':5,'授乳経験はありますか':5,
            '乳房に関する自覚症状':6,'自覚症状':6,'乳房自覚症状':6,'乳房に関する自覚症状はありますか':6,
            '月経開始':7,'MCD':7,'月経がある場合の最終月経':7,'月経周期':7,'月経について':7,'生理':7,
            '豊胸術':8,'豊胸手術の施行':8,'豊胸術の有無':8,
            '乳癌歴':9,'乳がんにおける家族歴':9,
            '家族の病気':10,'家族歴':10,'家族の方の病気':10,'がんに関する家族歴':10,
            '備考':11,'依頼医コメント':11,'依頼医からのコメント':11,'その他(気になる事など)':11,'【その他】':11,
            '※':12,'他覚的所見':12
            }

'''
rtemplate = {}
for k,v in template.items():
    if v not in rtemplate.keys():
        rtemplate[v] = []
    rtemplate[v].append(k)
'''

#　上記のtemplateからもらう番号で、項目を標準化する
dlist = {0:'依頼医',1:'技師名',2:'受診理由',3:'検診有無・異常所見',4:'既往歴',5:'授乳歴',6:'自覚症状',
         7:'MCD',8:'豊胸術',9:'乳癌歴',10:'癌に関する家族歴',11:'備考',12:'注意点'}


ptn_dict = {
    'date':re.compile('(?P<年>20[0-9]{1,2})(　| |年|/|\.|,)*(?P<月>[0-9]{0,2})'),  #日付のパターン
    'methods':re.compile('MMG|US|PET-CT'),                                        #診断方法のパターン
    'res':re.compile('結果（左右も）|結果（左右）|異常所見|結果|異常'),                  #結果のパターン
    'note':re.compile('左|右|両|乳房'),#注目するパターン
    
    'yes':re.compile('いる|異常有り|異常有|異常あり|あり\（続柄\）|あり\(続柄\)|有り|有|あり|ある'),
    'no':re.compile('異常なし|異常無し|異常無|無し|無い|無|なし|いない|ない'),
    'mb':re.compile('がんに関する記載なし|不明|記載無し|記載無|記載なし|記録なし|記録無し|記録無|わからない|\？|\?'),
    'family':re.compile('(祖父\(父方\)|祖父\(母方\)|祖母\(母方\)|祖母\(父方\)|両親の祖父|両親の祖母|祖父母|祖父|祖母|叔母|伯母|伯父|実母|母方のいとこ|母親|父親|父|母|兄弟姉妹|兄弟|姉妹|姉|妹|弟|兄|娘|息子|子|おば|おじ|長女|長男|いとこ)'),
    'sym':re.compile('( |　|:|：|・|【|】|\(|\)|（|）|/|／|、|,|→|に|;|；|\.|⇒)'),
    'cancer':re.compile('(子宮|癌|その他がん|がん|腫瘍|子宮頸|肺|大腸|胃|前立腺)'),
    'restall': ['y','n','m','s','-','c','r','nt']  #パターンの略称
    }


def heading(val,s2):
    
    if val==0:
        return re.match(' *(\(|（)?\w+：? ?(\(|（)?',s2)
    elif val==1:  #技師項目を検出するパターン
        return re.match(' *(\(|（)?\w+：? ?(\(|（)?(?P<技師>\w+)',s2)
    elif val==2:  #受信理由項目を検出するパターン
        return re.match('( |　|\n|[①-⑨]|【|\()*\w*理由( |　)*】?(:|：|】)?( |　|\n)*(\(|（|→)?( |　|\n)*',s2)
    elif val==3:  #検査歴
        return re.match('.*((診|査)歴|有無)( |　|:|：|\n|→|】)*',s2)
    elif val==4:  #既往歴
        return re.match('.*(歴|ありますか)( |　|:|：|\n|→|】)*',s2)
    elif val==5:  #授乳歴
        return re.match('.*授乳.*(歴|経験|は?ありますか)(年数)?( |　|:|：|\n|→|】)*',s2)
    elif val==6:  #自覚症状
        return re.match('.*自覚症状(は?ありますか)?( |　|:|：|\n|→|】)*',s2)
    elif val==7:  #MCD
        return re.match('( |　|\n|[①-⑨]|【|\()*(月経|MCD|開始|生理|周期)(について)?( |　|:|：|\n|→|】)*',s2)
    elif val==8:  #豊胸手術
        return re.match('( |　|\n|[①-⑨]|【|\()*豊胸手?術(の施行|の有無)?( |　|:|：|\n|→|】)*',s2)
    elif val==9:  #乳癌歴
        return re.match('( |　|\n|[①-⑨]|【|\()*(乳\w+歴)( |　|:|：|\n|→|】)*',s2)
    elif val==10: #家族歴
        return re.match('( |　|\n|[①-⑨]|【|\()*\w*家族\w*(歴|病気)( |　|:|：|\n|→|】)*',s2)
    elif val==11: #備考
        return re.match('( |　|\n|[①-⑨]|【|\()*(備考|依頼医\w+コメント)( |　|:|：|\n|→|】)*',s2)
        
# 文章にある各言葉を分類する関数
def getallstr(line, divlist):
    allstr = np.array(['-' for _ in range(len(line))]).astype('>U12')
    for div in divlist:
        if div in ['yes','no','mb','sym','cancer','res']:
            for i in ptn_dict[div].finditer(line): allstr[i.start():i.end()] = div[0];
        elif div in ['family','methods']:
            for i in ptn_dict[div].finditer(line): allstr[i.start():i.end()] = i.group();
        elif div in ['note']:
            for i in ptn_dict[div].finditer(line): allstr[i.start():i.end()] = 'nt';
        elif div == 'date':
            for i in ptn_dict['date'].finditer(line):
                m = i['月']
                y = i['年']
                if  y == '':
                    continue
                if m == '':
                    m = 6 
                allstr[i.start():i.end()] = y+'/'+str(m)
                
    return allstr

# 記載がない所に記載無しと記載する
def convnone(l):

    addlist = []
    npl = np.array(l).astype(str)

    for i in range(len(l)):
        addlist.append(l[i])
        for j in range(len(l[i])):
            if len(l[i])==3 and l[i][2]!='n' and l[i][j] == None:
                l[i][j] = '記載なし'
            if l[i][j] in ['その他がん','その他のがん','\t',' ','　']:
                l[i][j] = ' '
                
                if l[i][0] is not None and l[i][2] != 'n' and len(np.where(np.array(list(map(lambda x:(x[0] == l[i][0]) and (x[1] not in ['その他がん','その他のがん','\t',' ','　']) and (x[2] == l[i][2]),npl)))==True)[0]):
                    addlist = addlist[:-1]
              
    return addlist

# listをStringに変換する
def convltos(l): 
    st = ''
    for i in l:
        for j in i:
            if j == None:
                j = ' - '
            st = st +' '+ j
        st+='\n'
    return [st]
    
# 検出した項目の内容を更に細かい分割する
def processre(current,cstr,d,index,dlist,match,ptn_dict=None,ifprint=False):
    if ifprint:
        print(current,match)
    #依頼医
    if current == 0:
        None
        
    #技師名
    elif current == 1:
        if ifprint:
            print('項目　',dlist[current])
        
        if match:
            d[index][dlist[current]] = match['技師']
        else:
            print(current,'not aligned')
        if ifprint:
            print('回答',d[index][dlist[current]],'\n')
    
    #受診理由
    elif current == 2:
        if ifprint:
            print('項目　',dlist[current])
        
        match = re.match('( |　|\n|[①-⑨]|【|\()*\w*理由( |　)*】?(:|：|】)?( |　|\n)*(\(|（|→)?( |　|\n)*',cstr)
        if match:
            cstr = cstr[match.end():]
        else:
            print(current,'not aligned')
        if ifprint:
            print('回答',cstr,'\n')   
        d[index][dlist[current]] = cstr
        
    #検診有無・異常所見   -----------------------
    elif current == 3:
        if ifprint:
            print('項目　',dlist[current])

        if match:
            cstr = cstr[match.end():]
        else:
            print(current,'not aligned')
            
        # still left
        if ifprint:
            print('回答',cstr,'\n')
        
        #s = d[getrow(df,1250).index[0]]['検診有無・異常所見']
        metlist = []
        rlist = []
        for line in cstr.split('\n'):
            line = re.sub('(マンモグラフィー|マンモグラフィ|マンモ)','MMG',line)
            line = re.sub('超音波診断|超音波検査|超音波','US',line)
            #print(line)
            somestr = ''
            pval = [None,None,False]
            rval = [None,None]
            rflag = False
            if line == '':
                continue
            allstr = getallstr(line,['yes','no','mb','methods','sym','note','res','date'])
        
            #print(allstr)

            for j in range(len(allstr)):
        
                if allstr[j] not in ptn_dict['restall'] and allstr[j] != somestr:
                    somestr = allstr[j]
        
                    if re.match('[0-9]',somestr):
                        if pval[0] == 'y' or pval[0] is None:
                            pval[1] = somestr
        
                        else:
                            pval = [None,somestr,False]
        
                        if pval[2] == False:
        
                            for mi in reversed(range(len(metlist))):
                                if metlist[mi][1] is None and metlist[mi][2] is None:
                                    metlist[mi][1] = pval[1]
                                    metlist[mi][2] = pval[0]
                                elif metlist[mi][1] is None and metlist[mi][2] == 'y':
                                    metlist[mi][1] = pval[1]
                                elif metlist[mi][2] == 'n':
                                    break
                            if len(metlist)>0:
                                pval[2] = True
        
        
                    else:
                        if (len(metlist) == 0) or somestr not in [mindex[0] for mindex in metlist]:
        
                            metlist.append([somestr,None,None])
                            for mi in reversed(range(len(metlist))):
        
                                if metlist[mi][1] is None and metlist[mi][2] is None and pval[2] == False and(pval[0] is not None or pval[1] is not None):
        
                                    metlist[mi][1] = pval[1]
                                    metlist[mi][2] = pval[0]
                                elif metlist[mi][1] is None and metlist[mi][2] == 'y' and pval[1] is not None:
                                        metlist[mi][1] = pval[1]
        
                                elif metlist[mi][2] == 'n':
                                    break
        
                            if pval[1] is not None:
                                pval[2] = True
        
                elif allstr[j] == 'y' or allstr[j] == 'n' or allstr[j] == 'm' or allstr[j] == 'r' or allstr[j]== 'nt':
                    if (j==0 or (j>0 and allstr[j-1] != allstr[j])) and not rflag:
        
                        if allstr[j] in ['r','nt']:
                            rflag = True
                        elif allstr[j] == 'n':
                            pval = ['n',None,False]
                            for mi in reversed(range(len(metlist))):
        
                                if metlist[mi][1] is None and metlist[mi][2] is None:
                                    metlist[mi][2] = 'n'
                                else:
                                    break
                            pval[2] = True
                        else:
                            pval[0] = 'y'
                
                if rval[0] is None and (j>1 and allstr[j]!='r' and allstr[j-1]=='r'):
                    #print('r inside',j,allstr[j],line[j])
                    if allstr[j] == 's':
                        rval[0] = j+1
                    else:
                        rval[0] = j
                elif rval[0] is not None and allstr[j] == 's' and j == rval[0]:
                    rval[0] = j+1
            
                if allstr[j] == 'nt':
                    rval[1] = None
            
            if metlist == []:
                metlist.append([None,None,None])
            
            for mi in reversed(range(len(metlist))):
                if metlist[mi][1] is None and metlist[mi][2] is None and (pval[0] is not None or pval[1] is not None):
                    metlist[mi][1] = pval[1]
                    metlist[mi][2] = pval[0]
        
                elif metlist[mi][1] is None and metlist[mi][2] == 'y' and pval[1] is not None:
                    metlist[mi][1] = pval[1]
        
                elif metlist[mi][2] == 'n':
                    break
            
        
        
            if rval[0] is not None:
                if rval[1] is None:
                    rval[1] = len(line)  
        
                #print(line[rval[0]:rval[1]],rval[0],rval[1]) 
                rlist.append((line[rval[0]:rval[1]],allstr[rval[0]:rval[1]]))
        
        
            #print('')
        
        for i in range(len(metlist)):
            met,dt,yn = metlist[i]
            if met is not None and yn is None:
                metlist[i][2] = 'y'
            elif dt is None and yn is None:
                metlist[i][2] = 'n'
        #print(metlist)
        #print(rlist)
        
        
        mlist2 = np.array([[None,None],[None,None],[None,None]])
        p = 0
        desc = [None,None]
        methods = ['MMG','US','PET-CT']
        rline = []
        rorg = []
        for line,allstr in rlist:
            for i in range(len(allstr)):
                if allstr[i] in ['n','y','m']:
                    
                    if mlist2[p][1] is not None and (mlist2[p][0] is not None or mlist2[p][1]!=allstr[i]):
                        if p<2:
                            p+=1
                     
                    mlist2[p][1] = allstr[i]
                    if mlist2[p][0] is not None:
                        if p<2:
                            p+=1
                elif allstr[i] in methods and allstr[i] not in mlist2[:,0]:
                    
                    if mlist2[p][0] is not None:
                        if p<2:
                            p+=1
                        
                    mlist2[p][0] = allstr[i]
                    if mlist2[p][1] is not None:
                        if p<2:
                            p+=1
        
                if i<len(line)-1 and desc[0] is None and ((allstr[i] not in ['s','r','y','n','m'] and '-' in allstr[i:]) or allstr[i] == 'nt'):
                    desc[0] = i
                    desc[1] = len(line)
        
            if desc[0] is not None:
                #print(line[desc[0]:desc[1]])
                rline.append(line[desc[0]:desc[1]])
                rorg.append(allstr[desc[0]:desc[1]])
        npos = np.where((mlist2[:,0] == None) & (mlist2[:,1] == None))[0]
        if len(npos)>0:
            mlist2 = np.delete(mlist2,npos,0)
        
        if 'n' in mlist2[:,1] and None in mlist2[:,1]:
            mlist2[:,1] = 'n'
        elif 'y' in mlist2[:,1] and None in mlist2[:,1]:
            mlist2[:,1] = 'y'
        #print(mlist2)
        #print(rline)
        
        allcomp = {}
        allcomp['検診有無'] = convnone(metlist)
        allcomp['異常所見有無'] = mlist2
        allcomp['説明'] = rline
        d[index][dlist[current]] = allcomp
        
        
    #既往歴
    elif current == 4:
        if ifprint:
            print('項目　',dlist[current])
        
        if match:
            cstr = cstr[match.end():]
        else:
            print(current,'not aligned')
            
        # still left    まだ分割していない・必要に応じて分割を行う予定です。
        if ifprint:
            print('回答',cstr,'\n')
        d[index][dlist[current]] = cstr
    
    #授乳歴
    elif current == 5:
        if ifprint:
            print('項目　',dlist[current])
        
        if match:
            cstr = cstr[match.end():]
        else:
            print(current,'not aligned')
            
        # still left  まだ分割していない・必要に応じて分割を行う予定です。
        if ifprint:
            print('回答',cstr,'\n')
        
        l = getallstr(cstr,['yes','no','mb'])
        if len(l)>0:
            if 'y' in l or 'n' not in l:
                cstr = '有'
            if 'n' in l:
                cstr = '無'
        else:
            cstr = '記載無し'
        d[index][dlist[current]] = cstr
        
    #自覚症状
    elif current == 6:
        if ifprint:
            print('項目　',dlist[current])
       
        if match:
            cstr = cstr[match.end():]
        else:
            print(current,'not aligned')
         
        if ifprint:
            print('回答',cstr,'\n')  
        d[index][dlist[current]] = cstr

    #MCD
    elif current == 7:
        if ifprint:
            print('項目　',dlist[current])
        
        if match:
            cstr = cstr[match.end():]
        else:
            print(current,'not aligned')
            
        # still left  まだ分割していない・必要に応じて分割を行う予定です。
        if ifprint:
            print('回答',cstr,'\n')
        d[index][dlist[current]] = cstr

    #豊胸術
    elif current == 8:
        if ifprint:
            print('項目　',dlist[current])
        
        if match:
            cstr = cstr[match.end():]
        else:
            print(current,'not aligned')
            
        # still left  まだ分割していない・必要に応じて分割を行う予定です。
        if ifprint:
            print('回答',cstr,'\n')
        
        l = getallstr(cstr,['yes','no','mb'])
        if len(l)>0:
            if 'y' in l or 'n' not in l:
                cstr = '有'
            if 'n' in l:
                cstr = '無'
        else:
            cstr = '記載無し'
        d[index][dlist[current]] = cstr
    
    #乳癌歴
    elif current == 9:
        if ifprint:
            print('項目　',dlist[current])
        
        if match:
            cstr = cstr[match.end():]
        else:
            print(current,'not aligned')
        
        # still left
        if ifprint:
            print('回答',cstr,'\n')
        
        famlist = []
        for line in cstr.split('\n'):
            #print(line)
            start = 0
            end = 0
            famstr = ''
            cval = ['乳がん',None,False]
            if line == '':
                continue
            
            allstr = getallstr(line,['yes','no','mb','family','sym','cancer'])
            #print('all ',allstr)
            endp = np.where(allstr!='s')[0]
            if len(endp) == 0:
                continue
            endp = endp[-1]
            allstr = allstr[:endp+1]
            line = line[:endp+1]
        
            for j in range(len(allstr)):
                
                if allstr[j] not in ptn_dict['restall'] and (famstr!=allstr[j] or (famstr == allstr[j] and cval[0] != famlist[-1][1] and cval[2] == False)):
                    #print(2,'inside',j,allstr[j],line[j],cval,famstr,famlist)
                    famstr = allstr[j]
                    famlist.append([famstr,None,None])
                    
                    if cval[2] == False:
                        for fi in reversed(range(len(famlist))):
                            if famlist[fi][1] == None:
                                famlist[fi][1] = cval[0]
                                famlist[fi][2] = cval[1]
                                cval[2] = True
                            else:
                                break
                
                if allstr[j] in ['y','n','m']:
                    if j==0 or (j>0 and allstr[j-1] != allstr[j]):
                        cval[1] = allstr[j]
                        if cval[0] is not None and allstr[j] == 'n':
                            famlist.append([None,cval[0],'n'])
                            cval = ['乳がん',None,False]
                        
                        else:
                            for fi in reversed(range(len(famlist))):
                                if famlist[fi][1] == None:
                                    famlist[fi][1] = cval[0]
                                    famlist[fi][2] = cval[1]
                                    cval[2] = True
                                else:
                                    break
                                 
            for fi in reversed(range(len(famlist))):
                 if famlist[fi][1] == None:
                     famlist[fi][1] = cval[0]
                     famlist[fi][2] = cval[1]
                     cval[2] = True
                 else:
                     break
                 
        if cval[2] == False and cval[0] is not None or cval[1] is not None:
            if len(famlist)==0:
                famlist.append([None,cval[0],cval[1]])
            elif famlist[-1][1] != cval[0]:
                famlist.append([famlist[-1][0],cval[0],cval[1]])
                
        if len(famlist)>0:       
            famset = [famlist[0]]
            for i in range(1,len(famlist)):
                if famlist[i] != famlist[i-1]:
                    famset.append(famlist[i])
            famlist = famset
             
        for i in range(len(famlist)):
            if famlist[i][2] == None:
                famlist[i][2] = 'y'
        
        #print(famlist)
        d[index][dlist[current]] = convnone(famlist)
        
    #癌に関する家族歴    ---------------------
    elif current == 10:
        if ifprint:
            print('項目　',dlist[current])
        
        if match:
            cstr = cstr[match.end():]
        else:
            print(current,'not aligned')
        
        # still left
        if ifprint:
            print('回答',cstr,'\n')
        
        #s = d[getrow(df,1919).index[0]]['癌に関する家族歴']
        famlist = []
        clist = []
        for line in cstr.split('\n'):
            
            start = 0
            end = 0
            famstr = ''
            cval = [None,None,False]
            clist = []
        
            if line == '':
                continue
        
            allstr = getallstr(line,['yes','no','mb','family','sym','cancer'])
            
            endp = np.where(allstr!='s')[0]
            if len(endp) == 0:
                continue
            endp = endp[-1]
            allstr = allstr[:endp+1]
            line = line[:endp+1]
            #print('all ',allstr)
            
        
            for j in range(len(allstr)):
                if j> 0 and (allstr[j-1] == 'c' and allstr[j] != 'c') or (j==len(allstr)-1 and allstr[j] in ['-','c']):
                    #print(1,'inside',allstr[j],line[j],j,cval,famlist)
                    end = j
                    if j == len(allstr)-1 and allstr[j]!='s':
                        end = j+1
                    if cval[2] == True:
                          clist = []
        
                    if cval[1] is not None and cval[0] is None:
                        cval[0] = line[start:end]
                    else:
                        cval = [line[start:end],None,False]
        
                    clist.append(cval)
                    if cval[2] is False:
                        for fi in reversed(range(len(famlist))):
                            if famlist[fi][1] == None:
                                famlist[fi][1] = cval[0]
                                cval[2] = True
                            else:
                                break
        
                    start = j
        
                if allstr[j] != '-' and allstr[j] != 'c':
                    start = j+1
                    #print(j,start)
                if allstr[j] not in ptn_dict['restall'] and (famstr!=allstr[j] or (famstr == allstr[j] and cval[0] != famlist[-1][1] and cval[2] == False)):
                    #print(2,'inside',j,allstr[j],line[j],clist,famstr)
                    famstr = allstr[j]
                    famlist.append([famstr,None,None])
            
                    ccopy = clist.copy()
                    if len(clist)>0:
                        addlist = []
                        for cindex,c2 in enumerate(clist.copy()):
                            if c2[2] == False:
                                for fi in reversed(range(len(famlist))):
                                    if famlist[fi][1] == None:
                                        famlist[fi][1] = c2[0]
                                        famlist[fi][2] = c2[1]
                                        c2[2] = True
                                    else:
                                        break
                                if c2[2] == True:
                                    famlist.append([famstr,None,None])
                                else:
                                    addlist.append(cindex)
        
                        if famlist[-1][1] is None and famlist[-1][2] is None and clist[-1][2] == True and ccopy[-1][2] == False:
                            famlist = famlist[:-1]
        
                        cval = clist[-1]
                        ccopy = []
                        for cindex in addlist:
                            ccopy.append(clist[cindex])
        
                        clist = ccopy
        
                if allstr[j] in ['y','n','m']:
                    if j==0 or (j>0 and allstr[j-1] != allstr[j]):
                        cval[1] = allstr[j]
                        if cval[0] is not None and allstr[j] == 'n':
                            famlist.append([None,cval[0],'n'])
                            cval = [None,None,False]
        
                        elif cval[0] is not None and cval[2] == False:
                            for fi in reversed(range(len(famlist))):
                                if famlist[fi][1] == None:
                                    famlist[fi][1] = cval[0]
                                    famlist[fi][2] = cval[1]
                                    cval[2] = True
                                else:
                                    break
        
            for fi in reversed(range(len(famlist))):
                 if famlist[fi][1] == None:
                     famlist[fi][1] = cval[0]
                     famlist[fi][2] = cval[1]
                     cval[2] = True
                 else:
                     break
        
        if cval[2] == False and cval[0] is not None or cval[1] is not None:
            if len(famlist)==0:
                famlist.append([None,cval[0],cval[1]])
            elif famlist[-1][1] != cval[0]:
                famlist.append([famlist[-1][0],cval[0],cval[1]])
        
        if len(famlist)>0:       
            famset = [famlist[0]]
            for i in range(1,len(famlist)):
                if famlist[i] != famlist[i-1]:
                    famset.append(famlist[i])
            famlist = famset
        
        for i in range(len(famlist)):
            if famlist[i][2] is None:
                famlist[i][2] = 'y'
                
        #print(famlist)
        d[index][dlist[current]] = convnone(famlist)
    
    #備考
    elif current == 11:
        if ifprint:
            print('項目　',dlist[current])
        
        if match:
            cstr = cstr[match.end():]
        else:
            print(current,'not aligned')
        
        # still left 　　まだ分割していない・必要に応じて分割を行う予定です。
        if ifprint:
            print('回答',cstr,'\n')
        d[index][dlist[current]] = cstr
    
    #注意点
    elif current == 12:
        if ifprint:
            print('項目　',dlist[current])
        # まだ分割していない・必要に応じて分割を行う予定です。
    return d

#　検査目的の分割して、保存したデータを出力する　　　（上記のprocessre関数を用いる）
def create3(df):
    d = {}
    
    for index,s in enumerate(df['検査目的']):
        
        tk = list(template.keys())
        d[index] = {}
        d[index]['colid'] = df['colid'].iloc[index]
        dcopy = list(dlist.keys())
        
        s = s.replace('\u3000',' ')
        #print('-------- ',index)
        i = 0
        i2 = s[i:].find('\n')
        i3 = s[i:].find('\n\n')
        current = None
        current2 = None
        cstr = None
        inside = False
        match = None
        mcopy = None
        
        while i2>0 and i2<=len(s):
        
            s2 = s[i:i2]
            #print(i,i2,i3,current,current2,'s2-',s2)
            for tkey in tk:
                val = template[tkey]
                
                if tkey in s2 and val in dcopy:
                    mcopy = match
                    match = heading(val,s2)
                    if not match:
                        match = mcopy
                        mcopy = None
                        continue
                    
                    current2 = val
                    if current == None:
                        current = val
                        cstr = s2
        
                    dcopy.remove(val)
                    inside = True    
                    #print('--'+dlist[val],current,current2,cstr,match,mcopy)
                    break
                
            if not inside and current!= None:
                #print('not inside')
                cstr = cstr+'\n'+s2
        
            if (i2 == i3 or current2!=current) and current != None:
                #print('change ',cstr,current,current2)
                if mcopy == None or current==current2:
                    mcopy = match
                #print(match,mcopy)
                d = processre(current,cstr,d,index,dlist,mcopy,ptn_dict=ptn_dict,ifprint=False)
         
                if i2==i3:
                    if current2!=current:
                        d = processre(current2,s2,d,index,dlist,mcopy,ptn_dict=ptn_dict)
                        None
                    current = None
                    cstr = None
                    
                else:
                    current = current2 
                    cstr = s2
                    
                
            if i2==i3:
                i2 += 1
            
            i = i2+1
            if i >= len(s):
                #print('out3',mcopy,current,current2,cstr,s2)
                d = processre(current,cstr,d,index,dlist,match,ptn_dict=ptn_dict)
                break
            
            elif s[i:].find('\n')<0:
                i2 = len(s)
                
            else:
                i2 = i+s[i:].find('\n') 
                i3 = i+s[i:].find('\n\n')
            
            inside = False
            #print('final ',i,i2,i3,cstr)
            #print('         ***end***')
    
    # Merge 乳癌歴 and 家族歴　　　　（乳癌歴と家族歴の統合）
    for i in range(len(d)):
        k = d[i].keys()
        if '乳癌歴' in k:
            if '癌に関する家族歴' in k:
                d[i]['癌に関する家族歴'] = d[i]['乳癌歴']+d[i]['癌に関する家族歴']
            else:
                d[i]['癌に関する家族歴'] = d[i]['乳癌歴']
            del(d[i]['乳癌歴'])
    
        if '癌に関する家族歴' in k:
            addlist = []
            for j in range(len(d[i]['癌に関する家族歴'])):
                if d[i]['癌に関する家族歴'][j][2] != 'n':
                    addlist.append(d[i]['癌に関する家族歴'][j])
            d[i]['癌に関する家族歴'] = addlist
     
    return d
        


#データをexcelに保存する関数
def savedata(df,d1,d):
    dtotal = []
    for index in range(len(df)):
        s = 0
        
        if '受診理由' not in d[index].keys():
            print(index)
            s+=1
            dtotal.append([df['colid'].iloc[index],df['患者ID'].iloc[index],d1[index]['判定'],'記載無し'])
        
        elif '自覚症状' not in d[index].keys() or ('無' in d[index]['自覚症状'] or 'なし' in d[index]['自覚症状'] or 'ない' in d[index]['自覚症状']):
        
            dtotal.append([df['colid'].iloc[index],df['患者ID'].iloc[index],d1[index]['判定'],d[index]['受診理由']])
        
        else:
            dtotal.append([df['colid'].iloc[index],df['患者ID'].iloc[index],d1[index]['判定'],d[index]['受診理由']+'\n\n(自覚症状)\n'+d[index]['自覚症状']])
        
        if '癌に関する家族歴' in d[index].keys():
            dtotal[-1] = dtotal[-1] + convltos(d[index]['癌に関する家族歴'])
        else:
            dtotal[-1] = dtotal[-1] + [' - ']
        if '検診有無・異常所見' in d[index].keys():
            dtotal[-1] = dtotal[-1] + convltos(d[index]['検診有無・異常所見']['検診有無'])
            dtotal[-1] = dtotal[-1] + convltos(d[index]['検診有無・異常所見']['異常所見有無'])
            rline = d[index]['検診有無・異常所見']['説明']
            if len(rline) == 0:
                rline = [' - ']
            dtotal[-1] = dtotal[-1] + convltos(rline)
        else:
            for i in range(3):
                dtotal[-1] = dtotal[-1] + [' - ']
            
    from openpyxl import Workbook
    
    wb= Workbook()
    #ws=wb.active
    dt2 = pd.DataFrame(dtotal,columns=['colid','患者ID','判定','受診理由・自覚症状','家族歴','検診有無','異常所見有無','異常所見説明'])
    with pd.ExcelWriter(os.getcwd()+'\\Documents\\DWIBS\\分割3.xlsx') as writer:
        writer.book=wb
        dt2.to_excel(writer)
        writer.save()









