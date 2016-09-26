from __future__ import division

from nltk.tag import StanfordNERTagger
import nltk
import os
import urllib
from bs4 import BeautifulSoup
import math
import operator

NER_FOLDER="./stanford-ner-2015-04-20/classifiers/"
STANDFORD_FOLDER="./"
Pages=30

def search(query):
    # this Function get a query and search it in www.exalead.com ,then returns the 30 top retrieved urls
    url=url_maker(query)
    print "***** Saerching the following query:::: ",query.split()[0].replace("_"," ")
    print url
    try:
        html = urllib.urlopen(url).read()
    except:
        print "**** UNABLE TO OPEN THE REQUESTED PAGE ****"
        return []
    soup = BeautifulSoup(html, "html.parser")
    URLs=[]
    for line in str(soup).splitlines():
        if "<h4 class=\"media-heading\"><a class=\"title\" href=" in line :
            URLs.append(line.split("href=\"")[1].split("\" title")[0])
    return URLs

def get_plain_text(url):
    # this Function get the pure text of a html page
    try:
        html = urllib.urlopen(url).read()
    except:
        print "**** UNABLE TO OPEN THE REQUESTED PAGE ****"
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text

def url_maker(line):
    # this function make the URL to search the query in www.exalead.com

    NamedEntity=line.split()[0].replace("_"," ")
    Loc=line.split()[1:]
    Location=""
    for l in Loc:
        Location+=l.split(":")[0].replace("_"," ").strip()+" "

    query="http://www.exalead.com/search/web/results/?q="
    for n in NamedEntity.split():
        query+=n+" "

    # to include background location in the search uncomment the following lines

    # for l in Location.strip().split():
    #     query+=l.split(":")[0].replace("_"," ")+" "

    query=query.strip().replace(" ","+")+"&elements_per_page="+str(Pages)+"&start_index=0"
    return query

def Locations_in_text(text):
    # This Function Tag every word in a text and return the words with "Location" tag as a list

    st = StanfordNERTagger(NER_FOLDER+'english.all.3class.distsim.crf.ser.gz')

    TagedNamedEntity=st.tag(text.split())
    Locations=[]
    for NamedEntity in TagedNamedEntity:
        name=NamedEntity[0]
        tag=NamedEntity[1]
        if tag=="LOCATION":
            Locations.append(name)
    return Locations

def Pinh(Locations):
    # This function calculate Pinh(l|d,n) for all location in Locations and return a dictionary

    loca_tf=dict((loca,Locations.count(loca)) for loca in Locations)
    sum=0
    for loca in loca_tf:
        sum+=loca_tf[loca]
    return dict((location,loca_tf[location]/sum) for location in loca_tf)

def Indexes_located(seq,item):
    # This Function locate an item in a list and returns a list of indexes in which the item is located

    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

def Pdist(Locations,NamedEntity,DocumentText):
    # This function calculate Pnear(l|d,n) for all location in Locations and return a dictionary

    DocumentList=[word.lower().strip() for word in DocumentText.strip().split()]
    Locations=list(set(Locations))
    NamedEntityIndexes=Indexes_located(DocumentList,NamedEntity.lower().strip())
    print "\n               **** Indexes of Named entity ::: "
    print NamedEntity.lower().strip(),NamedEntityIndexes
    print "\n               **** Folowing is the indexes of each Location :::\n"
    dictionary={}
    for location in Locations:
        LocationIndexes=Indexes_located(DocumentList,location)
        print location ,LocationIndexes
        minimum_distance=DocumentList.__len__()+2
        for nind in NamedEntityIndexes:
            for lind in LocationIndexes:
                dist=abs(nind-lind)
                if dist<minimum_distance:
                    minimum_distance=dist
        dictionary[location]=minimum_distance

    denominator_sum=0
    for loca in dictionary:
        denominator_sum+=1/dictionary[loca]

    result={}
    for loca in dictionary:
        result[loca]=(1/dictionary[loca])/denominator_sum
    return result

def validate(Locations,Location):
    # This Function checks if the page contain the on of the background locations or not

    for part in str(Location).split():
        if part.strip().lower() in Locations:
            return True
    print "     **** The page does not contain any of the background locations *****"
    return False

def ShanonEntropy(dictionary):
    #This Function calculate shanon Entropy as mentioned in the paper

    sum=0
    for loca in dictionary:
        sum+=-1*(dictionary[loca]*math.log(dictionary[loca],10))

    Hmax=math.log(dictionary.__len__(),10)
    if dictionary.__len__()>1:
        return 1-(sum/Hmax)
    else :
        return 0

if __name__ == "__main__":

    os.environ["CLASSPATH"]=STANDFORD_FOLDER+"stanford-ner-2015-04-20/stanford-ner.jar"
    os.environ["STANFORD_MODELS"]=STANDFORD_FOLDER+"stanford-ner-2015-04-20/classifiers"


    NamedEntities=open("./NamedEntities/politicians.gt").readlines()
    for queryline in NamedEntities:
        NamedEntity=queryline.split()[0].split("_")[1]
        Loc=queryline.split()[1:]
        Location=""
        for l in Loc:
            Location+=l.split(":")[0].replace("_"," ").strip()+" "

        print "\n             **** Searching for NamedEntity :::",NamedEntity
        RetreivedURLs=search(queryline)
        if RetreivedURLs!=[]:
            count=1
            for url in RetreivedURLs:
                print "\n*****Getting data from this page ::::"
                print "Page number "+str(count)+" :"+url
                DocumentText=get_plain_text(url)

                if DocumentText!="":

                    Locations=Locations_in_text(DocumentText.strip())
                    Locations_lower=[]
                    for loca in Locations:
                        try:
                            Locations_lower.append(str(loca).lower())
                        except:
                            continue

                    if validate(Locations_lower,Location):

                        loca_P_dict_inh=Pinh(Locations_lower)
                        loca_P_dict_dist=Pdist(Locations_lower,NamedEntity,DocumentText)
                        print "\n             *****result******       \n"
                        print "**** inheritance probability ::" , loca_P_dict_inh
                        print "**** distance probability :::",loca_P_dict_dist
                        J=ShanonEntropy(loca_P_dict_dist)
                        print "\n**** Shanon Entropy      J(d,n) :::",J

                        combined_loca_p=dict((loca , (loca_P_dict_dist[loca]*J+(1-J)*loca_P_dict_inh[loca])) for loca in loca_P_dict_dist)
                        sorted_combine = sorted(combined_loca_p.items(), key=operator.itemgetter(1))


                        print "**** combined probability :::",sorted_combine

                        print "\n             ***** Following is the predicted geo-center of the Named entity by this page ******       \n"
                        print "                           >>>>Predicted Location:::",sorted_combine[-1]


                    # break
                print "\n###########   Next Page     #############"
                count+=1
        print "\n#######################\n#######################\n##### NEXT QUERY ######\n#######################\n#######################"
        # break
