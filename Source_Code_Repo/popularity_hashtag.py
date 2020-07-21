


import pickle
import pandas
import networkx as nx
from nltk import *
import nltk
def oauth_login():
    CONSUMER_KEY = 'OCa2LGsxL0EUALx6zRUjQWeHl'
    CONSUMER_SECRET = 'JWCtpW8inPkfC6QUJbtfJ9uz02JcO78dC5sJDi4obx5LZcBCc5'
    OAUTH_TOKEN = '1100042597370920961-4kzA5Em8CbPk4q8jE6GwnSXp3gSdyS'
    OAUTH_TOKEN_SECRET = '3fAlogX64SO98ZWyQLAaE4ok7AHljlEHp8r3QcWqNWkM7'
    
    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY, CONSUMER_SECRET)
    
    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api


import sys
import time
from urllib.error import URLError
from http.client import BadStatusLine
import twitter

def make_twitter_request(twitter_api_func, max_errors=10, *args, **kw):
    # A nested helper function that handles common HTTPErrors. Return an updated
    # value for wait_period if the problem is a 500 level error. Block until the
    # rate limit is reset if it's a rate limiting issue (429 error). Returns None
    # for 401 and 404 errors, which requires special handling by the caller.
    def handle_twitter_http_error(e, wait_period=2, sleep_when_rate_limited=True):

        if wait_period > 3600:  # Seconds
            print('Too many retries. Quitting.', file=sys.stderr)
            raise e

        # See https://developer.twitter.com/en/docs/basics/response-codes
        # for common codes

        if e.e.code == 401:
            print('Encountered 401 Error (Not Authorized)', file=sys.stderr)
            return None
        elif e.e.code == 404:
            print('Encountered 404 Error (Not Found)', file=sys.stderr)
            return None
        elif e.e.code == 429:
            print('Encountered 429 Error (Rate Limit Exceeded)', file=sys.stderr)
            if sleep_when_rate_limited:
                print("Retrying in 15 minutes...ZzZ...", file=sys.stderr)
                sys.stderr.flush()
                time.sleep(60 * 15 + 5)
                print('...ZzZ...Awake now and trying again.', file=sys.stderr)
                return 2
            else:
                raise e  # Caller must handle the rate limiting issue
        elif e.e.code in (500, 502, 503, 504):
            print('Encountered {0} Error. Retrying in {1} seconds' \
                  .format(e.e.code, wait_period), file=sys.stderr)
            time.sleep(wait_period)
            wait_period *= 1.5
            return wait_period
        else:
            raise e

    # End of nested helper function

    wait_period = 2
    error_count = 0

    while True:
        try:
            return twitter_api_func(*args, **kw)
        except twitter.api.TwitterHTTPError as e:
            error_count = 0
            wait_period = handle_twitter_http_error(e, wait_period)
            if wait_period is None:
                return
        except URLError as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print("URLError encountered. Continuing.", file=sys.stderr)
            if error_count > max_errors:
                print("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise
        except BadStatusLine as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print("BadStatusLine encountered. Continuing.", file=sys.stderr)
            if error_count > max_errors:
                print("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise




from functools import partial
from sys import maxsize as maxint
def get_friends_followers_ids(twitter_api, screen_name=None, user_id=None,
                              friends_limit=maxint, followers_limit=maxint):
    # Must have either screen_name or user_id (logical xor)
    assert (screen_name != None) != (user_id != None), \
        "Must have screen_name or user_id, but not both"

    # See http://bit.ly/2GcjKJP and http://bit.ly/2rFz90N for details
    # on API parameters

    get_friends_ids = partial(make_twitter_request, twitter_api.friends.ids,
                              count=100)
    get_followers_ids = partial(make_twitter_request, twitter_api.followers.ids,
                                count=100)

    friends_ids, followers_ids = [], []

    for twitter_api_func, limit, ids, label in [
        [get_friends_ids, friends_limit, friends_ids, "friends"],
        [get_followers_ids, followers_limit, followers_ids, "followers"]
    ]:

        if limit == 0: continue

        cursor = -1
        while cursor != 0:

            # Use make_twitter_request via the partially bound callable...
            if screen_name:
                response = twitter_api_func(screen_name=screen_name, cursor=cursor)
            else:  # user_id
                response = twitter_api_func(user_id=user_id, cursor=cursor)

            if response is not None:
                ids += response['ids']
                cursor = response['next_cursor']



            # XXX: You may want to store data during each iteration to provide an
            # an additional layer of protection from exceptional circumstances

            if len(ids) >= limit or response is None:
                break

    # Do something useful with the IDs, like store them to disk...
    return friends_ids[:friends_limit], followers_ids[:followers_limit]




def crawl_followers(twitter_api, screen_name, first_connection, user_id,limit=5000 ):

    next_queue=first_connection

    nodes = [user_id]
    nodes.extend(first_connection)
    edge=list(zip([0]*len(first_connection),first_connection))
    while len(nodes) <= limit:

        (queue, next_queue) = (next_queue, [])

        for fid in queue:
            friends_ids, follower_ids = get_friends_followers_ids(twitter_api, user_id=fid)
            connection = []
            connection.extend(friends_ids)
            connection.extend(follower_ids)


            nodes.extend(connection)
            edge.extend(list(zip([fid]*len(connection),connection)))
            nodes.extend(connection)
            # print(f"{fid}'s connection are {connection}")
            next_queue.extend(connection)


            if len(nodes)>limit:
                break
    return edge, nodes

def BMP(s):
    return "".join((i if ord(i) < 10000 else '\ufffd' for i in s))

def get_my_object(twitter_api, screen_names=None, user_ids=None):
    # Must have either screen_name or user_id (logical xor)
    assert (screen_names != None) != (user_ids != None), \
        "Must have screen_names or user_ids, but not both"

    items_to_info = []

    if screen_names:
        response = make_twitter_request(twitter_api.users.lookup,
                                        screen_name=screen_names)
    else:
        response = make_twitter_request(twitter_api.users.lookup,
                                        user_id=user_ids)
    #print(response)

    for user_info in response:


        items_to_info.append(str(user_info['id']))

        items_to_info.append(user_info['screen_name'])
        print(user_info['screen_name'])
        if 'status' in user_info:
            content=user_info['status']['text']
            content=BMP(content)
            print(content)
            contentgbk=content.encode('gbk','ignore');
            content=contentgbk.decode('gbk','ignore');
            
            items_to_info.append(content)

        if user_info['location'] == '':

            items_to_info.append('')
        else:
            items_to_info.append(user_info['location'])

        items_to_info.append(str(user_info['followers_count']))
        items_to_info.append(str(user_info['friends_count']))



    return items_to_info

def store_data(screen_name,limit):

    premium_search_args = oauth_login()

    response = make_twitter_request(premium_search_args.users.lookup,
                         screen_name=screen_name)
    user_id = ''
    for r in response:
        user_id = r['id']
    friends_ids, followers_ids = get_friends_followers_ids(premium_search_args, screen_name=screen_name,
                                                           friends_limit=100,
                                                           followers_limit=100)
    connection = []
    connection.extend(friends_ids)
    connection.extend(followers_ids)
    edge,nodes = crawl_followers(premium_search_args, screen_name, connection, user_id,limit=limit)
    print("edge")
    print(edge)
    print(nodes)
    all_user_info = []
    with open('data.csv', 'w',encoding='utf8') as file:
        for e in set(nodes):
            user_info = get_my_object(premium_search_args, user_ids=e)
            all_user_info.append(user_info)
            
        df = pandas.DataFrame(all_user_info,columns=['id','name','text','location','friend_count','follower_count'])
        pandas.DataFrame.to_csv(df,file,index=None,encoding="utf-8")
    with open('edge.pickle','wb') as file:
        pickle.dump(edge, file)

def load_edge():
    with open('edge.pickle','rb') as file:
        edges = pickle.load(file)
    return edges
# load data from the file
# x:id y:screen name z:text content w:number of friends  v:number of followers
#
#
def load_inf():
    df=pandas.read_csv('data.csv',header=None,sep=',')
    x=df[0][1:]
    y=df[1][1:]
    z=df[2][1:]
    w=df[4][1:]
    v=df[5][1:]
    #print (x)
    return x,y,z,w,v
import math
import matplotlib.pyplot as plt

# calculate the centrality of every node in the graph
def centrality():
    edge = load_edge()
    G=nx.Graph()
    G.add_edges_from(edge)
    nodes=G.node
    #print(nodes)
    cloness=nx.closeness_centrality(G)
    cloness=dict(sorted(cloness.items(),key=lambda item:item[1],reverse=True))
    degree=nx.degree_centrality(G)
    betweenness=nx.betweenness_centrality(G)
    del cloness[0]
    #print(cloness)
    return cloness,degree,betweenness
    
def networkplot():
    edge = load_edge()


    G = nx.Graph()
    G.add_edges_from(edge)
    print("the diameter is  ", nx.diameter(G))
    print("the average distance is  ", nx.average_shortest_path_length(G))
    print('number of nodes:', G.number_of_nodes())
    print('number of edges:', G.number_of_edges())
    nx.draw(G, with_labels=False, font_weight='bold')
    plt.savefig("final_edge.png")
    plt.show()

#find every hashtag after "#"  
def My_Filter(mystr):

    tt=TweetTokenizer(mystr)
    tokens = tt.tokenize(mystr)
    tokens_filtered = [w for w in tokens if w[0]=='#']  ####
    for w in range(0,len(tokens_filtered)):
        tokens_filtered[w]=tokens_filtered[w].lower()
    length_filtered = [w for w in tokens_filtered if len(w)>=2]
    
    #print(length_filtered)
    result=length_filtered
    """
    porter = PorterStemmer()
    stems = [porter.stem(t) for t in tokens]
    #print(stems)

    from nltk.corpus import stopwords
    import string
    stop=stopwords.words('english')
    #print(stop)
    
    tokens_filtered = [w for w in stems if w.lower() not in stop and w.lower() not in string.punctuation]
    for w in range(1,len(tokens_filtered)):
        tokens_filtered[w]=tokens_filtered[w].lower()

    length_filtered = [w for w in tokens_filtered if len(w)>=4]

    taged_sent = nltk.pos_tag(length_filtered)
    taged_sent=dict(taged_sent)
    deleteset=["CC","CD","EX","IN","JJ","MD","UH","VB"]
    postag_filtered=[w for w in length_filtered if (not(taged_sent[w] in deleteset))]
    special=[]
    special=["today","love","happi"]
    result=[w for w in postag_filtered if (w not in special)]
    #print(length_filtered)
    #print(taged_sent)
    """
    return result
#get the most common hashtags from the community    
def getcommonword(x,y,z):
    combine=[];
    allwords=[];
    for i in range(1,z.shape[0]):
        mystr=""
        t=type(0.0)
        if type(z[i])==t:
            mystr=""
            
        else:
            mystr=z[i]
            non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
            mystr=mystr.translate(non_bmp_map)
            #print(mystr)
            
        words=My_Filter(mystr)       
        allwords.append(words)
        for j in range(0,len(words)):
            combine.append(words[j])
            
         
    #print(allwords)    
        
    fdist = FreqDist(combine)
    
    
    #print("fdist")
    #print(fdist.most_common(3))
  
    popular3=[]
    for k in range(0,3):
        popular3.append(fdist.most_common(100)[k][0])
    
    #print(popular3)
    choose1=[]
    choose2=[]
    choose3=[]
    for k in range(0,len(allwords)):
        for n in range(0,len(popular3)):
            if popular3[n] in allwords[k]:
                choose1.append(x[k+1])
                choose2.append(y[k+1])
                choose3.append(allwords[k])
                
    #print(choose1)
    #print(choose2)
    #print(choose3)
    return popular3
    #length_filtered = [w for w in tokens_filtered if len(w)>=3]
# The main function to do recommended based on friends and followers/closeness centrality/hashtag
def jqpart():
    x,y,z,w,v=load_inf()
    xx=list(map(int, x))
    d=zip(xx,y)
    id_name=dict(d)
    d=zip(xx,z)
    id_text=dict(d)
    for i in range(1,len(w)):
        if (math.isnan(float(w[i]))):
            w[i]='0'
    for i in range(1,len(v)):
        if (math.isnan(float(v[i]))):
            v[i]='0'        
        
    ww=list(map(int, w))
    
    d=zip(xx,ww)
    id_friends=dict(d)
    
    vv=list(map(int, v))
    d=zip(xx,vv)
    id_followers=dict(d)
    screen_name=[]

    popular3=getcommonword(x,y,z)
    print("The most popular three words=")
    print(popular3)

    #user_id,screen_name=getcommonword(x,y,z)
    #user_id = list(map(int, user_id))

    closeness,degree,betweenness=centrality()
    #print("closeness")
    #print(closeness)
    #print("id_name")
    #print(id_name)

    
    #p = {key:value for key, value in closeness.items() if key in user_id}
    #p = {key:value for key, value in degree.items() if key in user_id}
    #p = {key:value for key, value in betweenness.items() if key in user_id}
    #print(user_id)
    #print(p)
    
    id_friends=dict(sorted(id_friends.items(),key=lambda item:item[1],reverse=True))
    #print(id_friends)
    id_followers=dict(sorted(id_followers.items(),key=lambda item:item[1],reverse=True))
    #sorted(closeness.items(),key=lambda item:item[1])
    n=3
    result1=[]
    result2=[]
    result3=[]
    print("Most friends")  
    for i in range(0,n):
        k = list(id_friends.keys())[i]
        j=i+1
        print("No."+str(j))
        print("User_ID="+str(k))
        print("closeness="+str(closeness[k]))
        print("ScreenName="+id_name[k])
        print("friends="+str(id_friends[k]))
        print("followers="+str(id_followers[k]))
        non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
        idstr=id_text[k].translate(non_bmp_map)
        print("His text="+idstr)
        result1.append(k)
    print("Most followers")    
    for i in range(0,n):
        k = list(id_followers.keys())[i]
        j=i+1
        print("No."+str(j))
        print("User_ID="+str(k))
        print("closeness="+str(closeness[k]))
        print("ScreenName="+id_name[k])
        print("friends="+str(id_friends[k]))
        print("followers="+str(id_followers[k]))
        non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
        idstr=id_text[k].translate(non_bmp_map)
        print("His text="+idstr)
        result2.append(k)    
    print("Centrality")
    for i in range(0,n):
        k = list(closeness.keys())[i]
        j=i+1
        print("No."+str(j))
        print("User_ID="+str(k))
        print("closeness="+str(closeness[k]))
        print("ScreenName="+id_name[k])
        print("friends="+str(id_friends[k]))
        print("followers="+str(id_followers[k]))
        non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
        idstr=id_text[k].translate(non_bmp_map)
        print("His text="+idstr)
        result3.append(k)   

"""
    screen_name = 'stacynance85'
    node_limit = 1000
    store_data(screen_name,node_limit)
    networkplot()
"""

if __name__=='__main__':
    #networkplot()
    jqpart()




    
