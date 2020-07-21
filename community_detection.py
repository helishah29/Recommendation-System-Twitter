import networkx.algorithms.community as cm
import pickle
import community
from community import community_louvain
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy
from scipy.cluster import hierarchy
from scipy.spatial import distance
import random
from sklearn.cluster import SpectralClustering
from collections import defaultdict
import pandas as pd
def networkplot():
    with open('edge.pickle', 'rb') as file:
        edges = pickle.load(file)
    G = nx.Graph()
    # G.add_edges_from([(1,2),(1,3),(1,4),(2,3),(3,4),(4,5),(4,6),(5,6),(5,7),(5,8),(6,5),(6,8),(7,9),(7,8),(7,6),(9,10),(10,11),(11,12)])
    G.add_edges_from(edges)
    l = nx.find_cliques(G)
    # l = cm.k_clique_communities(G,3)
    # for ll in l:
    #     print(ll)
    # print(cm.k_clique_communities(G,3))


    # better with karate_graph() as defined in networkx example.
    # erdos renyi don't have true community structure
    # G = nx.erdos_renyi_graph(30, 0.05)

    # first compute the best partition
    partition = community.best_partition(G)
    # print(partition)

    # dendrogram = community_louvain.generate_dendrogram(G)
    # # print(dendrogram)
    # print(community.partition_at_level(dendrogram,level = 0))
    # drawing
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(G)
    # print(pos)
    count = 0
    node_color = ['r','y','c']
    louvain_community = []
    for com in set(partition.values()):


        list_nodes = [nodes for nodes in partition.keys()
                      if partition[nodes] == com]
        louvain_community.append(list_nodes)

        nx.draw_networkx_nodes(G, pos, list_nodes, node_size=20,
                               node_color=node_color[count])
        count += 1
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    # nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()
    return louvain_community

def gather_tweet():
    louvain_community = spectral_cluster()
    with open('data.csv','r') as file:
        df = pd.read_csv(file)
    ct = []
    for n in louvain_community:
        # print(df[df['id'].astype('int').isin(n)]['text'].fillna('').values)
        community_text = df[df['id'].astype('int').isin(n)]['text'].fillna('').values
        ct.append(community_text)
    with open('spectral_ct.pickle','wb') as file:
        pickle.dump(ct,file)



def lda():

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation


    with open('louvain_ct.pickle', 'rb') as f:
        result_set = pickle.load(f)[1]
        print(result_set)
    # json.dumps(result_set,'result.json')
    # for r in result_set[:3]:
    #     print(r)
    # print(result_set[:3])

    # for file in os.listdir('./110-f-d'):
    #     with open(os.path.join('./110-f-d',file),'r') as f:
    #         print(f.read().TEXT)
    # def chinese_word_cut(mytext):
    #     return " ".join(jieba.cut(mytext))
    #
    # df["content_cutted"] = df.content.apply(chinese_word_cut)
    #
    #
    n_features = 1000

    '''
    min_df: When building the vocabulary ignore terms that have a document
    frequency strictly lower than the given threshold.
    '''
    tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                    max_features=n_features,
                                    stop_words='english',
                                    max_df=0.5,
                                    min_df=10, ngram_range=(1, 2)
                                    )

    tf = tf_vectorizer.fit_transform(result_set)
    # tf: (docid, keyword_id), freq
    print(tf)

    n_topics = 5
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                    learning_method='online', learning_offset=50., random_state=0)
    lda.fit(tf)
    # topic distribution in each doc, shape:(doc length, n_topics)
    print(lda.fit_transform(tf))

    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            # topic shape: (1000,)
            # topic: the key words distribution in each doc
            print("topic", topic.shape)
            print("Topic #%d:" % topic_idx)
            print(" ".join(feature_names[i]
                           for i in topic.argsort()[:-n_top_words - 1:-1]))
        print()

    # 20 key words for each topictf_feature_names = tf_vectorizer.get_feature_names()
    n_top_words = 20
    # the 1000 key words
    tf_feature_names = tf_vectorizer.get_feature_names()
    print(len(tf_feature_names))
    print_top_words(lda, tf_feature_names, n_top_words)

    import pyLDAvis.sklearn
    data = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
    pyLDAvis.show(data)

def create_hc(G):
    """Creates hierarchical cluster of graph G from distance matrix"""
    path_length = nx.all_pairs_shortest_path_length(G)
    distances = numpy.zeros((len(G), len(G)))
    for u, p in path_length:
        for v, d in p.items():
            distances[u][v] = d
    # Create hierarchical cluster
    Y = distance.squareform(distances)
    Z = hierarchy.complete(Y)  # Creates HC using farthest point linkage
    # This partition selection is arbitrary, for illustrive purposes
    membership = list(hierarchy.fcluster(Z, t=1.15))
    # Create collection of lists for blockmodel
    partition = defaultdict(list)
    for n, p in zip(list(range(len(G))), membership):
        partition[p].append(n)
    print(partition)
    return list(partition.values())


def spectral_cluster():
    with open('edge.pickle', 'rb') as file:
        edges = pickle.load(file)
    G = nx.Graph()
    # edges = [(1,2),(1,3),(1,4),(2,3),(3,4),(4,5),(4,6),(5,6),(5,7),(5,8),(6,5),(6,8),(7,9),(7,8),(7,6)]
    # G.add_edges_from([(1,2),(1,3),(1,4),(2,3),(3,4),(4,5),(4,6),(5,6),(5,7),(5,8),(6,5),(6,8),(7,9),(7,8),(7,6)])
    G.add_edges_from(edges)
    print(edges[0])
    # G.add_edges_from(edges)
    # G.add_edges_from(edges[:100])
    # nx.draw(G, node_color='y', with_labels=True, node_size = 10)
    X = numpy.array(edges)
    clustering = SpectralClustering(n_clusters=2,assign_labels = "discretize",random_state = 0).fit_predict(X)

    l=defaultdict(list)

    for i in range(len(X)):
        l[clustering[i]].extend(X[i])
    pos = nx.spring_layout(G)
    print(l.keys())
    node_color = ['r','c','y','c']
    count = 0
    community_node = []
    for k,v in l.items():
        # print(type(k))
        # print(type(v))
        list_nodes = list(set(v))
        community_node.append(list_nodes)
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size=5,
                               node_color=node_color[count])
        count += 1
    # nx.draw_networkx_edge_labels(G, pos, alpha=0.5)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    # nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()
    # print(clustering)
    return community_node
# networkplot


def block_model():
    with open('edge.pickle', 'rb') as file:
        edges = pickle.load(file)

    G = nx.Graph()
    G.add_edges_from([(1,2),(1,3),(1,4),(2,3),(3,4),(4,5),(4,6),(5,6),(5,7),(5,8),(6,5),(6,8),(7,9),(7,8),(7,6)])
    # nx.draw(G)
    # Extract largest connected component into graph H
    H = next(nx.connected_component_subgraphs(G))
    # Makes life easier to have consecutively labeled integer nodes
    H = nx.convert_node_labels_to_integers(H)

    # Create parititions with hierarchical clustering
    partitions = create_hc(H)
    # Build blockmodel graph
    BM = nx.quotient_graph(H, partitions, relabel=True)

    # Draw original graph
    pos = nx.spring_layout(H, iterations=100)
    plt.subplot(211)
    nx.draw(H, pos, with_labels=False, node_size=10)

    # Draw block model with weighted edges and nodes sized by number of internal nodes
    node_size = [BM.nodes[x]['nnodes'] * 10 for x in BM.nodes()]
    edge_width = [(2 * d['weight']) for (u, v, d) in BM.edges(data=True)]
    # Set positions to mean of positions of internal nodes from original graph
    posBM = {}
    node_color = ['r', 'b','y','c']
    count = 0
    for n in BM:
        print(n)
        xy = numpy.array([pos[u] for u in BM.nodes[n]['graph']])
        print(xy)
        posBM[n] = xy.mean(axis=0)
        list_nodes = partitions[count]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size=5,
                               node_color=node_color[count])
        count += 1
    plt.subplot(212)
    nx.draw(BM, posBM, node_size=node_size, width=edge_width, with_labels=False)
    plt.axis('off')
    plt.show()

# networkplot()
# spectral_cluster()
# block_model()
# gather_tweet()
lda()