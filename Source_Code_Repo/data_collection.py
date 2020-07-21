import pickle
import pandas
import networkx as nx
def oauth_login():
    CONSUMER_KEY = 'Gzy6AjP6DDIbtDWxf94Ucc3Nk'
    CONSUMER_SECRET = 'kRMAj0U87ayefou6dHJfgIsScnCq79mmruRNaztGvpDm8VoY4u'
    auth = twitter.oauth2.OAuth2(CONSUMER_KEY,CONSUMER_SECRET,'AAAAAAAAAAAAAAAAAAAAADc99gAAAAAAzNRZe6QmvPvfKjwAImrzpmQECHc%3DbjCdwhOCbBg0N8TUcywwhi7jpFK1UtSC3M09H0slzo5vIz93mP')
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
                              count=20)
    get_followers_ids = partial(make_twitter_request, twitter_api.followers.ids,
                                count=20)

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
            friends_ids = friends_ids[:10]
            follower_ids = follower_ids[:10]
            connection = []
            connection.extend(friends_ids)
            connection.extend(follower_ids)


            nodes.extend(connection)
            edge.extend(list(zip([fid]*len(connection),connection)))
            nodes.extend(connection)
            # print(f"{fid}'s connection are {connection}")
            next_queue.extend(connection)
            print(len(nodes))

            if len(nodes)>limit:
                break
    return edge, nodes


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
    # print(response)

    for user_info in response:


        items_to_info.append(str(user_info['id']))

        items_to_info.append(user_info['screen_name'])
        print(user_info['screen_name'])
        if 'status' in user_info:
            items_to_info.append(user_info['status']['text'])

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
                                                           friends_limit=20,
                                                           followers_limit=20)
    connection = []
    connection.extend(friends_ids)
    connection.extend(followers_ids)
    edge,nodes = crawl_followers(premium_search_args, screen_name, connection, user_id,limit=limit)
    all_user_info = []
    with open('data2.csv', 'w') as file:
        for e in set(nodes):
            user_info = get_my_object(premium_search_args, user_ids=e)
            all_user_info.append(user_info)

        df = pandas.DataFrame(all_user_info,columns=['id','name','text','location','friend_count','follower_count'])
        pandas.DataFrame.to_csv(df,file,index=None)
    with open('edge2.pickle','wb') as file:
        pickle.dump(edge, file)

def load_edge():
    with open('edge2.pickle','rb') as file:
        edges = pickle.load(file)
    return edges

def get_community_data(t):
    # friends_C, followers_C = get_friends_followers_ids(t,
    #                                                    screen_name='CongressListing', friends_limit=10000,
    #                                                    followers_limit=0)
    # friends_D, followers_D = get_friends_followers_ids(t, screen_name='TheDemocrats',
    #                                                    friends_limit=10000, followers_limit=0)
    # DC = [val for val in friends_D if val in friends_C]  # filtering
    # friends_R, followers_R = get_friends_followers_ids(t, screen_name='GOP',
    #                                                    friends_limit=10000, followers_limit=0)
    # RC = [val for val in friends_R if val in friends_C]  # filtering
    # G = nx.Graph()
    # G.add_nodes_from(DC)
    # G.add_nodes_from(RC)
    # with open('G.pickle','wb') as file:
    #     pickle.dump(G,file)
    # nx.draw(G, with_labels=False, font_weight='bold')
    # plt.savefig("final_edge3.png")
    with open('G.pickle','rb') as file:
        G = pickle.load(file)
    try:
        for id in G.nodes():
            friends, followers = get_friends_followers_ids(t, user_id=id,
                                                           friends_limit=10000, followers_limit=0)
            for f in friends:
                # only add edges to nodes already in G
                # no need to add_node(); no need to check duplicates
                if f in G.nodes():
                    G.add_edge(id, f)
    except:  # protected users will be skipped
        pass
    nx.write_adjlist(G, "assignment3_2018.adjlist")
    with open('G2.pickle', 'wb') as file:
        pickle.dump(G, file)
    nx.draw(G, with_labels=False, font_weight='bold')
    plt.savefig("community.png")
    plt.show()

import matplotlib.pyplot as plt
def networkplot():
    edge = load_edge()


    G = nx.Graph()
    G.add_edges_from(edge)
    print("the diameter is  ", nx.diameter(G))
    print("the average distance is  ", nx.average_shortest_path_length(G))
    print('number of nodes:', G.number_of_nodes())
    print('number of edges:', G.number_of_edges())
    nx.draw(G, with_labels=False, font_weight='bold')
    plt.savefig("final_edge2.png")
    plt.show()

if __name__=='__main__':
    # premium_search_args = oauth_login()
    # get_community_data(premium_search_args)
    screen_name = 'stacynance85'
    node_limit = 10000
    store_data(screen_name,node_limit)
    networkplot()