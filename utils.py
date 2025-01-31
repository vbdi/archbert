##### ni
import networkx as nx
import py_stringmatching
from sentence_transformers import util

def jaccard_similarity(g, h):
    i = set(g).intersection(h)
    return round(len(i) / (len(g) + len(h) - len(i)),3)

########
def select_k(spectrum, minimum_energy = 0.9):
    running_total = 0.0
    total = sum(spectrum)
    if total == 0.0:
        return len(spectrum)
    for i in range(len(spectrum)):
        running_total += spectrum[i]
        if running_total / total >= minimum_energy:
            return i + 1
    return len(spectrum)

# Another method is to use what is called Eigenvector Similarity. Basically, you calculate the Laplacian eigenvalues for the adjacency matrices of each of the graphs. For each graph, find the smallest k such that the sum of the k largest eigenvalues constitutes at least 90% of the sum of all of the eigenvalues. If the values of k are different between the two graphs, then use the smaller one. The similarity metric is then the sum of the squared differences between the largest k eigenvalues between the graphs. This will produce a similarity metric in the range [0, âˆž), where values closer to zero are more similar.
def Eigenvector_Similarity(G1,G2):
    laplacian1 = nx.spectrum.laplacian_spectrum(G1.to_undirected())
    laplacian2 = nx.spectrum.laplacian_spectrum(G2.to_undirected())

    k1 = select_k(laplacian1)
    k2 = select_k(laplacian2)
    k = min(k1, k2)

    similarity = sum((laplacian1[:k] - laplacian2[:k])**2)

    lap_cos_score = util.cos_sim(laplacian1[:k], laplacian2[:k])

    return similarity, lap_cos_score

def Cos_score(node_feats_1, node_feats_2, shape_inds_1, shape_inds_2, _Adjs_1, _Adjs_2):
    min_len = min([len(node_feats_1), len(node_feats_2)])
    cos_node_score = py_stringmatching.similarity_measure.cosine.Cosine().get_sim_score(
        list(node_feats_1[:min_len]), list(node_feats_2[:min_len]))
    cos_shape_score = py_stringmatching.similarity_measure.cosine.Cosine().get_sim_score(
        list(shape_inds_1[:min_len].view(-1)), list(shape_inds_2[:min_len].view(-1)))
    cos_edge_score = py_stringmatching.similarity_measure.cosine.Cosine().get_sim_score(
        list(_Adjs_1[:min_len, :min_len].reshape(-1)), list(_Adjs_2[:min_len, :min_len].reshape(-1)))
    cos_score = (cos_node_score + cos_shape_score + cos_edge_score) / 3.0

    return cos_score



def get_scores(G1, G2, node_feats_1, node_feats_2, shape_inds_1, shape_inds_2, _Adjs_1, _Adjs_2):
    # jaccard_score: [0,1], 1: similar, 0: not similar
    jaccard_score = (jaccard_similarity(G1.nodes(), G2.nodes()) + jaccard_similarity(G1.edges(), G2.edges())) / 2.0

    # Graph edit distance is a graph similarity measure analogous to Levenshtein distance for strings. It is defined as minimum cost of edit path (sequence of node and edge edit operations) transforming graph G1 to graph isomorphic to G2.
    # edit_dis = nx.graph_edit_distance(G1, G2)
    eigen_dis, lap_cos_score = Eigenvector_Similarity(G1, G2)
    #cos_score = py_stringmatching.similarity_measure.cosine.Cosine().get_sim_score([1],[2])
    # too slow!
    #tacsim_score = tacsim_combined(G1, G2).mean()

    cos_score = Cos_score(node_feats_1, node_feats_2, shape_inds_1, shape_inds_2, _Adjs_1, _Adjs_2)

    return jaccard_score, eigen_dis, cos_score #tacsim_score#, cos_score