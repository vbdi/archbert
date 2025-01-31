#sa: This file includes the functions for the visualizations
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from sentence_transformers import util
from matplotlib.cm import get_cmap

class visualization_class:
    def __init__(self,visualization_mode):
        self.param = 0
        self.visualization_counter = -1
        self.visualization_mode =visualization_mode
        if visualization_mode == 1:
            self.all_features = []
            self.all_texts = []
            self.all_types = []
        if visualization_mode == 0:
            self.all_ax = []
    def main_visualize(self,scores,langModel,arch_emb,query_emb,text,names,counter):
        one_index = np.where(np.array(scores) == 1.0)
        query_token_emb = langModel.encode(sentences=text, batch_size=len(text), convert_to_tensor=True,
                                           output_value='token_embeddings')
        if len(one_index[0]) > 2:
            # if first_figure:
            # f, ax = plt.subplots()
            # first_figure = False
            #f, ax = plt.subplots() #vis mode 1
            all_f=[]
            if self.visualization_mode == 0 and counter in [357,425,449]:
                f, ax = plt.subplots(1,2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})
                self.visualization_counter += 1
                ax[0], arhc_and_text = self.visualize_arch_text(arch_emb, query_emb, text, names, counter, one_index,
                                                        ax[0],
                                                        mode=self.visualization_mode,
                                                        vis_counter=self.visualization_counter)
                handles, labels = ax[0].get_legend_handles_labels()
                ax[1].set_axis_off()
                legend = ax[1].legend(handles, labels, loc='center left', bbox_to_anchor=(0.02, 0.5), fontsize=10,
                                      borderaxespad=0.)
                # self.all_ax.append(ax[0])
                # #all_ax.append(ax[1])
                # if counter==449:
                #     f, ax = plt.subplots(3, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})
                #     for i, axis in enumerate(ax.ravel()):
                #         if i==0:
                #             axis=self.all_ax[0]
                #             handles, labels = self.all_ax[0].get_legend_handles_labels()
                #         elif i==1:
                #             legend = axis.legend(handles, labels)
                #
                #         if i==2:
                #             axis=self.all_ax[1]
                #             handles, labels = self.all_ax[1].get_legend_handles_labels()
                #         elif i==3:
                #             legend = axis.legend(handles, labels)
                #
                #         if i==4:
                #             axis=self.all_ax[2]
                #             handles, labels = self.all_ax[2].get_legend_handles_labels()
                #         elif i==5:
                #             legend = axis.legend(handles, labels)

                    #legend = ax[1].legend(handles, labels,loc='center left', bbox_to_anchor=(0.02, 0.5), fontsize=10, borderaxespad=0.)
                f.savefig('./visualize_sa/arch_vs_text_for_one' + str(counter) + '.png',bbox_inches='tight')#,
            elif self.visualization_mode == 1 and (counter in [187, 271, 284,357,448,449]): #
                self.all_features, self.all_texts, self.all_types = self.gather_embedding(arch_emb, query_emb, text, names, counter,
                                                                      one_index,
                                                                      self.all_features, self.all_texts, self.all_types)
                ax = self.visualize_all(self.all_features, self.all_texts, self.all_types, ax,counter)
                ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, borderaxespad=0.)
                f.savefig('./visualize_sa/arch_vs_text_for_multiple' + str(counter) + '.png',
                          bbox_inches='tight')
            # plt.show()
        # for token embeddings and confusion matrix
        # for one_single_index in  one_index[0]:
        #     query_tokens_number = query_token_emb[one_single_index].shape[0]
        #     if query_tokens_number<50:
        #         text_for_arch = text[one_single_index]
        #         text_token_embd = query_token_emb[one_single_index]
        #         arch_token_embd = all_archs_emb[counter][0]
        #         arch_name=names[counter]
        #         #visualize_arhc_text_tokens(text_token_embd,arch_token_embd,counter, text_for_arch, arch_name)
        #         confiusion_arhc_text_tokens(text_token_embd, arch_token_embd, counter, text_for_arch,
        #                                    arch_name)
        # visualize_confusion()

    def visualize_arch_text(self,arch_emb, query_emb,text, names, counter,score_one_index,ax,mode=0,vis_counter=0):
        '''Function for visualizing the embeddings of Arch Vs a embeddings of different texts '''
        #==== Normalization
        #max_arch= torch.max(arch_emb)
        #min_arch = torch.min(arch_emb)
        #max_query = torch.max(query_emb)
        #min_query = torch.min(query_emb)
        #if max_arch>max_query:
        #    max_all= max_arch
        #else:
        #    max_all = max_query

        #if min_arch<min_query:
        #    min_all= min_arch
        #else:
        #    min_all = min_query

        #normalize vectors:
        #query_emb_norm = (query_emb - min_all)/(max_all - min_all)
        #arch_emb_norm = (arch_emb - min_all) / (max_all - min_all)
        arch_name  = names[counter]
        #return
        arch_and_positive_texts = []


        all_features=[]
        all_features.append(arch_emb.cpu().numpy())
        for query_one in query_emb:
            all_features.append(query_one.cpu().numpy())
        #all_features.append(query_emb_norm)
        text_all = []
        usefulname = arch_name.split("___")[1]
        usefulname =  usefulname.replace("_", "-")
        #all_texts.append(r"$\bf{Architecture: }$"+ usefulname)
        text_all.append(r"$\bf{Architecture: }$"+ usefulname)
        for text_one in text:
            text_all.append(text_one)

        #dim_reducer = TSNE(n_components=2)
        dim_reducer = PCA(n_components=2)
        features_dim_reduced_vectors = dim_reducer.fit_transform(all_features)
        #plt.scatter(features_dim_reduced_vectors[0, 0], features_dim_reduced_vectors[0, 1],c="g")
        if mode==0:
            #colors = cm.rainbow(np.linspace(0, 1, features_dim_reduced_vectors.shape[0]))
            name = "tab20"
            cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
            colors = cmap.colors  # type: list
        elif mode==1:
            name = "tab20"
            cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
            colors = cmap.colors  # type: list
        #fig = plt.figure()
        for ind_leg,(x,y) in enumerate(zip(features_dim_reduced_vectors[:,0], features_dim_reduced_vectors[:,1])): #colors
            if mode==0:        #mode=0: We have positive and negative samples in the visualization
                if ind_leg==0:
                    marker = "d"
                    marker_size = 121
                elif ind_leg-1 in np.array(score_one_index):
                    marker ="+"
                    marker_size = 121
                else:
                    marker = "X" #X
                    marker_size = 100
                if len(text_all[ind_leg].split(" "))<20 and ind_leg<20:
                    legend_text = " ".join(text_all[ind_leg].split(" ")[:7]) if len(text_all[ind_leg].split(" ")) > 10 else text_all[ind_leg]
                    ax.scatter(x, y, color=colors[ind_leg],label=legend_text,marker=marker,s=144)

        #plt.scatter(features_dim_reduced_vectors[1:,0], features_dim_reduced_vectors[1:,1])

        #ax.set_title('Compare embeddings for ' + arch_name.split("___")[1])
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=8,borderaxespad=0.) #
        #plt.tight_layout(pad=10)
        #fig.savefig('./visualize_sa/arch_vs_text_'+str(counter)+'.png', bbox_inches='tight')
        #plt.show()

        return ax, arch_and_positive_texts

    def visualize_all(self,all_features, all_texts,all_types,ax,counter=0):
        '''Function for visualizing the embeddings of Arch Vs a embeddings of different texts '''
        #dim_reducer = TSNE(n_components=2)
        dim_reducer = PCA(n_components=2)
        features_dim_reduced_vectors = dim_reducer.fit_transform(all_features)
        #plt.scatter(features_dim_reduced_vectors[0, 0], features_dim_reduced_vectors[0, 1],c="g")
        name = "tab20"
        cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
        colors = cmap.colors  # type: list
        #colors = cm.rainbow(np.linspace(0, 1, 50))
        #fig = plt.figure()
        for ind_leg,(x,y) in enumerate(zip(features_dim_reduced_vectors[:,0], features_dim_reduced_vectors[:,1])): #colors
            if all_types[ind_leg] == 0:
                marker = "d"
                color_arch = colors[ind_leg]
                if counter==449 and (ind_leg==7): #set manually
                    color_arch = colors[6] #set to green
                elif counter==449 and (ind_leg==13):
                    color_arch = colors[4] #set to red
                #if ind_leg%2==0:
                #    color_arch = colors[ind_leg]
                #else:
                #    color_arch = colors[(-1*ind_leg)]
                ax.scatter(x, y, color=color_arch, label=all_texts[ind_leg], marker=marker,s=121)
            else:
                #if ind_leg%3==0:
                marker = "+"
                #elif ind_leg%3==1:
                #    marker = "*"
                #else:
                #    marker = "."
                if counter==449 and (ind_leg==17):
                    legend_text = " ".join(all_texts[ind_leg].split(" ")[:14])
                elif counter==449 and (ind_leg==1):
                    legend_text = " ".join(all_texts[ind_leg].split(" ")[:7])
                elif counter==449 and (ind_leg==11):
                    all_texts[ind_leg] = all_texts[ind_leg].replace('1x1 Convolution, ','')
                    all_texts[ind_leg] = all_texts[ind_leg].replace('Convolution, ', '')
                    all_texts[ind_leg] = all_texts[ind_leg].replace('Dropout, ', '')
                    all_texts[ind_leg] = all_texts[ind_leg].replace('Batch Normalization, ', '')
                    legend_text = " ".join(all_texts[ind_leg].split(" ")[:11])
                else:
                    legend_text = all_texts[ind_leg] #" ".join(all_texts[ind_leg].split(" ")[:7]) if len(all_texts[ind_leg].split(" "))>  10 else all_texts[ind_leg]
                if counter==449 and (ind_leg==4 or ind_leg==6): #one point was annoying and ugly
                    continue
                else:
                    ax.scatter(x, y, color=color_arch, label=legend_text, marker=marker,s=144)
        return ax

    def gather_embedding(self,arch_emb, query_emb,text, names, counter,score_one_index, all_features=None, all_texts=None,all_types=None):
        '''Function for visualizing the embeddings of Arch Vs a embeddings of different texts '''
        arch_name  = names[counter]
        print("the counter is",counter)
        arch_name  = names[counter]
        #return
        all_features.append(arch_emb.cpu().numpy())
        usefulname = arch_name.split("___")[1]
        usefulname =  usefulname.replace("_", "-")
        all_texts.append(r"$\bf{Architecture: }$"+ usefulname)

        all_types.append(0)
        for i, query_one in enumerate(query_emb):
            if (i) in np.array(score_one_index) and  len(text[i].split(" "))<20: #20
                all_features.append(query_one.cpu().numpy())
                all_texts.append(text[i])
                all_types.append(1)
        return all_features,all_texts,all_types

    def visualize_arhc_text_tokens(self,text_token_embd,arch_token_embd,counter,text_for_arch, arch_name):
        '''Function for visualizing the TOKEN embeddings of Arch Vs a TOKEN embeddings of the same texts '''
        all_tokens=[]
        for text_token_embd_one in text_token_embd:
            all_tokens.append(text_token_embd_one.cpu().numpy())

        for arch_token_embd_one in arch_token_embd:
            all_tokens.append(arch_token_embd_one.cpu().numpy())
        #dim_reducer = TSNE(n_components=2)
        dim_reducer = PCA(n_components=2)
        features_dim_reduced_vectors = dim_reducer.fit_transform(all_tokens)
        #plt.scatter(features_dim_reduced_vectors[0, 0], features_dim_reduced_vectors[0, 1],c="g")

        #colors = cm.rainbow(np.linspace(0, 1, features_dim_reduced_vectors.shape[0]))
        fig = plt.figure()
        for ind_leg,(x,y) in enumerate(zip(features_dim_reduced_vectors[:,0], features_dim_reduced_vectors[:,1])):
            if ind_leg<len(text_token_embd):
                marker = "x"
                c='r'
            else:
                marker = 'd'
                c='b'
            plt.scatter(x, y, color=c,marker=marker) #label=text_all[ind_leg]

        #plt.suptitle('token embeddings of an arch: {} vs {}'.format( arch_name.split("___")[1],text_for_arch))
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=8,borderaxespad=0.) #
        #plt.tight_layout(pad=10)
        fig.savefig('./visualize_sa/arch_vs_text_tokens_'+str(counter)+'.png', bbox_inches='tight')
        #plt.show()

    def confiusion_arhc_text_tokens(self,text_token_embd,arch_token_embd,counter,text_for_arch, arch_name):
        '''Function for visualizing Confiusion mattix between
        the TOKEN embeddings of Arch Vs a TOKEN embeddings of the same texts '''
        text_all_tokens=[]
        arch_all_tokens=[]
        for text_token_embd_one in text_token_embd:
            text_all_tokens.append(text_token_embd_one)

        for arch_token_embd_one in arch_token_embd:
            arch_all_tokens.append(arch_token_embd_one)

        confusion_matrix = np.zeros((len(text_all_tokens),len(arch_all_tokens)))
        for i, text_token in enumerate(text_all_tokens):
            for j, arch_token in enumerate(arch_all_tokens):
                confusion_matrix[i,j] = (util.cos_sim(text_token, arch_token)[:, 0]).cpu()

        fig = plt.figure()
        im = plt.imshow(confusion_matrix, aspect='auto')
        fig.colorbar(im, orientation='vertical')
        fig.savefig('./visualize_sa/confisuion_'+str(counter)+'.png', bbox_inches='tight')
        #plt.show()


def AR_baseline_score(text,names,counter,all_scores):

    arch_name = names[counter]
    arch_name = arch_name.split("___")[1]
    archname_splits = arch_name.split("-")
    score_this_call = []

    for text_one in text:
        score_postive_flag = False
        counter_splits = 0.
        for  archname_split in archname_splits:
            if  archname_split in text_one:
                counter_splits +=1.
                all_scores.append(1.0)
                score_postive_flag = True
                break
        if  score_postive_flag==False:
            all_scores.append(0.0)
    return all_scores