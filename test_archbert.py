import na_search
import na_reasoning
import na_clone_detection
import bimodal_na_clone_detection
import na_qa
import clone_na_search
import bimodal_clone_na_search
import decoder
import streamlit as st

import numpy as np
import argparse

from pprint import pprint

def run_task(task='search', demo = True, validate = True, batch_size = 1, dataset = 'tvhf', num_nets = 100, roma=False,
             checkpoint_epoch=None,hypernet='gat',langModel = None,archModel = None,cross_encoder = True,
             max_node_size = 512, max_edge_size = 512,layernorm=False,
             data_path="", modelpath="",qa_mode='multi',visualize=False):

    if task=='search':
        if demo:
            st.title("ArchBERT - Architecture Search")
            query = st.text_input("Search query:", "object detectors using resnet").lower()
        else:
            query = "object detectors using resnet"

        if len(query)>0:
            doc_score_pairs = na_search.validate(langModel=langModel,
                                                 archModel=archModel,
                                                 dataset=dataset,
                                                 data_path=data_path,
                                                 num_nets=num_nets,
                                                 batch_size=batch_size,
                                                 cross_encoder=cross_encoder,
                                                 modelpath=modelpath,
                                                 hypernet=hypernet,
                                                 max_node_size=max_node_size,
                                                 max_edge_size=max_edge_size, query=query, checkpoint_epoch=checkpoint_epoch)

            # Output archs & scores
            top=200
            for doc, score in doc_score_pairs:#[:top]:
                if demo:
                    st.markdown('%.3f | %s' % (score,doc))
                else:
                    print(score, doc)
    elif task=='reasoning':
        if demo:
            st.title("ArchBERT - Architectural Reasoning")

        if validate: # run numerical analysis
            acc, f1 = na_reasoning.validate(langModel=langModel,
                                            archModel=archModel,
                                            dataset=dataset,
                                            data_path=data_path,
                                            num_nets=num_nets,
                                            batch_size=batch_size,
                                            cross_encoder=cross_encoder,
                                            modelpath=modelpath,
                                            hypernet=hypernet,
                                            max_node_size=max_node_size,
                                            max_edge_size=max_edge_size, checkpoint_epoch=checkpoint_epoch,
                                            visualize=visualize)
        else: # just run the demo
            acc=0.0
            f1=0.0
            if demo:
                query = st.text_input("Input statement:", "a residual network").lower()
            else:
                query = "a residual network"

            score = na_reasoning.demo(langModel=langModel,
                                      archModel=archModel,
                                      dataset=dataset,
                                      data_path=data_path,
                                      num_nets=num_nets,
                                      batch_size=batch_size,
                                      cross_encoder=cross_encoder,
                                      modelpath=modelpath,
                                      hypernet=hypernet,
                                      max_node_size=max_node_size,
                                      max_edge_size=max_edge_size, query=query, checkpoint_epoch=checkpoint_epoch)

        if demo:
            st.markdown('Dataset: %s' % ('AutoNet' if dataset=='autonet' else 'TVHF'))
            if num_nets:
                st.markdown('#Samples: %d' % (num_nets))
            else:
                st.markdown('#Samples: ALL')

            if validate:
                st.markdown('Accuracy: %.3f' % (acc))
                st.markdown('F1: %.3f' % (f1))
            else:
                st.markdown('The statement is: **%s**' % ('Correct.' if np.round(score) else 'Incorrect.'), unsafe_allow_html=False)
        else:
            print('Dataset: %s' % ('AutoNet' if dataset=='autonet' else 'TVHF'))
            if num_nets:
                print('#Samples: %d' % (num_nets))
            else:
                print('#Samples: ALL')
            if validate:
                print("Accuracy: "+str(acc))
                print("F1: " + str(f1))
            else:
                print('The statement is: %s' % ('Correct.' if np.round(score) else 'Incorrect.'))

        return acc, f1
    elif task=='na_clone_detection':
        if demo:
            st.title("ArchBERT - Architecture Clone Detection")

        if dataset == 'autonet':
            st.markdown('na_clone_detection for this dataset is not implmeneted yet!')
            print('na_clone_detection for this dataset is not implmeneted yet!')
        else:
            acc=0.0
            f1=0.0
            if validate: # run numerical analysis
                acc, f1, jacard_acc, jacard_f1 = na_clone_detection.validate(langModel=langModel,
                                                                             archModel=archModel,
                                                                             dataset=dataset,
                                                                             data_path=data_path,
                                                                             num_nets=num_nets,
                                                                             batch_size=batch_size,
                                                                             cross_encoder=cross_encoder,
                                                                             modelpath=modelpath,
                                                                             hypernet=hypernet,
                                                                             max_node_size=max_node_size,
                                                                             max_edge_size=max_edge_size, checkpoint_epoch=checkpoint_epoch)
            else: # just run the demo
                sim_score = na_clone_detection.demo(langModel=langModel,
                                                    archModel=archModel,
                                                    dataset=dataset,
                                                    data_path=data_path,
                                                    num_nets=num_nets,
                                                    batch_size=batch_size,
                                                    cross_encoder=cross_encoder,
                                                    modelpath=modelpath,
                                                    hypernet=hypernet,
                                                    # setting this to None gives poor resutls for this task!
                                                    # by "None" we mean padding with max-size in that batch (see grpah.py - batch_grpah_padding)
                                                    max_node_size=max_node_size, max_edge_size=max_edge_size, checkpoint_epoch=checkpoint_epoch
                                                    )

            if demo:
                st.markdown('Dataset: %s' % ('AutoNet' if dataset=='autonet' else 'TVHF'))
                if num_nets:
                    st.markdown('#Samples: %d' % (num_nets))
                else:
                    st.markdown('#Samples: ALL')
                if validate:
                    st.markdown('Accuracy: %.3f' % (acc))
                    st.markdown('F1: %.3f' % (f1))
                    st.markdown('Jacard Accuracy: %.3f' % (jacard_acc))
                    st.markdown('Jacard F1: %.3f' % (jacard_f1))
                else:
                    st.markdown('The architectures are: **%s**' % ('Similar.' if np.round(sim_score) else 'NOT Similar.'), unsafe_allow_html=False)
            else:
                print('Dataset: %s' % ('AutoNet' if dataset=='autonet' else 'TVHF'))
                if num_nets:
                    print('#Samples: %d' % (num_nets))
                else:
                    print('#Samples: ALL')
                if validate:
                    print("Accuracy: "+str(acc))
                    print("F1: " + str(f1))
                    print("Jacard Accuracy: "+str(jacard_acc))
                    print("Jacard F1: " + str(jacard_f1))
                else:
                    print('The architectures are: %s' % ('Similar.' if np.round(sim_score) else 'NOT Similar.'))
        return acc, f1
    elif task=='bimodal_na_clone_detection':
        if demo:
            st.title("ArchBERT - BiModal Architecture Clone Detection")

        if dataset == 'autonet':
            st.markdown('bimodal_na_clone_detection for this dataset is not implmeneted yet!')
            print('bimodal_na_clone_detection for this dataset is not implmeneted yet!')
        else:
            if validate: # run numerical analysis
                acc, f1 = bimodal_na_clone_detection.validate(langModel=langModel,
                                                              archModel=archModel,
                                                              dataset=dataset,
                                                              data_path=data_path,
                                                              num_nets=num_nets,
                                                              batch_size=batch_size,
                                                              cross_encoder=cross_encoder,
                                                              modelpath=modelpath,
                                                              hypernet=hypernet,
                                                              # setting this to a fixed padding size (e.g., 512) gives poor resutls for this task!
                                                              # by "None" we mean padding with max-size in that batch (see grpah.py - batch_grpah_padding)
                                                              max_node_size=max_node_size, max_edge_size=max_edge_size, checkpoint_epoch=checkpoint_epoch)
            else: # just run the demo
                acc = 0.0
                f1 = 0.0
                if demo:
                    query = st.text_input("Query:", "a residual network").lower()
                else:
                    query = "a residual network"

                sim_score = bimodal_na_clone_detection.demo(langModel=langModel,
                                                            archModel=archModel,
                                                            dataset=dataset,
                                                            data_path=data_path,
                                                            num_nets=num_nets,
                                                            batch_size=batch_size,
                                                            cross_encoder=cross_encoder,
                                                            modelpath=modelpath,
                                                            hypernet=hypernet,
                                                            max_node_size=max_node_size,
                                                            max_edge_size=max_edge_size, query=query, checkpoint_epoch=checkpoint_epoch)

            if demo:
                st.markdown('Dataset: %s' % ('AutoNet' if dataset=='autonet' else 'TVHF'))
                if num_nets:
                    st.markdown('#Samples: %d' % (num_nets))
                else:
                    st.markdown('#Samples: ALL')
                if validate:
                    st.markdown('Accuracy: %.3f' % (acc))
                    st.markdown('F1: %.3f' % (f1))
                else:
                    st.markdown('The architectures are: **%s**' % ('Similar.' if np.round(sim_score) else 'NOT Similar.'), unsafe_allow_html=False)
            else:
                print('Dataset: %s' % ('AutoNet' if dataset=='autonet' else 'TVHF'))
                if num_nets:
                    print('#Samples: %d' % (num_nets))
                else:
                    print('#Samples: ALL')
                if validate:
                    print("Accuracy: "+str(acc))
                    print("F1: " + str(f1))
                else:
                    print('The architectures are: %s' % ('Similar.' if np.round(sim_score) else 'NOT Similar.'))
        return acc, f1
    elif task=='clone_na_search':
        if demo:
            st.title("ArchBERT - Clone Architecture Search")
            st.markdown('Dataset: %s' % ('AutoNet' if dataset=='autonet' else 'TVHF'))
            st.markdown('#Samples: %d' % (num_nets))
        else:
            print('Dataset: %s' % ('AutoNet' if dataset=='autonet' else 'TVHF'))
            print('#Samples: %d' % (num_nets))

        if validate:
            st.markdown('Numerical analysis for this task is not implemented yet!')
            print('Numerical analysis for this task is not implemented yet!')
        else:
            arch_query_name, doc_score_pairs = clone_na_search.demo(langModel=langModel,
                                                                    archModel=archModel,
                                                                    dataset=dataset,
                                                                    data_path=data_path,
                                                                    num_nets=num_nets,
                                                                    batch_size=batch_size,
                                                                    cross_encoder=cross_encoder,
                                                                    modelpath=modelpath,
                                                                    hypernet=hypernet,
                                                                    max_node_size=max_node_size,
                                                                    max_edge_size=max_edge_size, checkpoint_epoch=checkpoint_epoch)
            # Output archs & scores
            top=200
            for doc, score in doc_score_pairs:#[:top]:
                if demo:
                    st.markdown('%.3f | %s' % (score,doc))
                else:
                    print('%.3f | %s' % (score,doc))
    elif task=='bimodal_clone_na_search':
        if demo:
            st.title("ArchBERT - Bi-Modal Clone Architecture Search")
            st.markdown('Dataset: %s' % ('AutoNet' if dataset=='autonet' else 'TVHF'))
            st.markdown('#Samples: %d' % (num_nets))
        else:
            print('Dataset: %s' % ('AutoNet' if dataset=='autonet' else 'TVHF'))
            print('#Samples: %d' % (num_nets))

        if validate:
            print('TBD')
        else:
            if demo:
                query = st.text_input("Search query:", "object detection").lower()
            arch_query_name, doc_score_pairs = bimodal_clone_na_search.demo(langModel=langModel,
                                                                            archModel=archModel,
                                                                            dataset=dataset,
                                                                            data_path=data_path,
                                                                            num_nets=num_nets,
                                                                            batch_size=batch_size,
                                                                            cross_encoder=cross_encoder,
                                                                            modelpath=modelpath,
                                                                            hypernet=hypernet,
                                                                            max_node_size=max_node_size,
                                                                            max_edge_size=max_edge_size, query=query.strip(), checkpoint_epoch=checkpoint_epoch)
            # Output archs & scores
            top = 200
            for doc, score in doc_score_pairs:#[:top]:
                if demo:
                    st.markdown('%.3f | %s' % (score,doc))
                else:
                    print('%.3f | %s' % (score,doc))
    elif task=='qa':
        qa_mode = 'multi'
        qa_modelpath = modelpath#+ "_qa" + ("_" + qa_mode if qa_mode == "multi" else "")

        qaModel = None
        if demo:
            st.title("ArchBERT - Architectural Question Answering")
            st.markdown('Dataset: %s' % ('AutoNet' if dataset=='autonet' else 'TVHF'))
            if num_nets:
                st.markdown('#Samples: %d' % (num_nets))
            else:
                st.markdown('#Samples: ALL')
        else:
            print('Dataset: %s' % ('AutoNet' if dataset=='autonet' else 'TVHF'))
            if num_nets:
                print('#Samples: %d' % (num_nets))
            else:
                print('#Samples: ALL')
        if dataset == 'tvhf':
            st.markdown('QA for this dataset is not implmeneted yet!')
        else:
            if validate:
                f1, acc = na_qa.validate(langModel=langModel,
                                         archModel=archModel,
                                         qaModel=qaModel,
                                         dataset=dataset,
                                         data_path=data_path,
                                         num_nets=num_nets,
                                         batch_size=batch_size,
                                         cross_encoder=cross_encoder,
                                         qa_modelpath=qa_modelpath,
                                         hypernet=hypernet,
                                         max_node_size=max_node_size,
                                         max_edge_size=max_edge_size, mode=qa_mode,
                                         checkpoint_epoch=checkpoint_epoch)
            else:
                predicted_answers, actual_answers, f1, acc = na_qa.demo(langModel=langModel,
                                                                        archModel=archModel,
                                                                        qaModel=qaModel,
                                                                        dataset=dataset,
                                                                        data_path=data_path,
                                                                        num_nets=num_nets,
                                                                        batch_size=batch_size,
                                                                        cross_encoder=cross_encoder,
                                                                        qa_modelpath=qa_modelpath,
                                                                        hypernet=hypernet,
                                                                        max_node_size=max_node_size,
                                                                        max_edge_size=max_edge_size, mode=qa_mode, checkpoint_epoch=checkpoint_epoch)
                if demo:
                    st.markdown('Predicted Answers: \r\n' + str(predicted_answers).replace('*','x'))
                    st.markdown('Actual Answers: \r\n' + str(actual_answers).replace('*', 'x'))
                else:
                    print('Predicted Answers: \r\n' + str(predicted_answers))
                    print('Actual Answers: \r\n' + str(actual_answers))

            if demo:
                st.markdown('F1 Score: %.3f' % (f1))
                st.markdown('Accuracy: %.3f' % (acc))
            else:
                print("F1 Score: "+str(f1))
                print("Accuracy: " + str(acc))
        return acc, f1
    elif task=='langdec':
        langdec_modelpath = modelpath
        langDecModel = None
        if demo:
            st.title("ArchBERT - Architecture Captioning")
            st.markdown('Dataset: %s' % ('AutoNet' if dataset=='autonet' else 'TVHF'))
            if num_nets:
                st.markdown('#Samples: %d' % (num_nets))
            else:
                st.markdown('#Samples: ALL')
        else:
            print('Dataset: %s' % ('AutoNet' if dataset=='autonet' else 'TVHF'))
            if num_nets:
                print('#Samples: %d' % (num_nets))
            else:
                print('#Samples: ALL')
        if dataset == '1ours':
            st.markdown('Arch2Lang for this dataset is not implmeneted yet!')
        else:
            if validate:
                some_score = 0.0
                bleu_score, rouge_score = decoder.validate(langModel=langModel,
                                                           archModel=archModel,
                                                           langDecModel=langDecModel,
                                                           dataset=dataset,
                                                           data_path=data_path,
                                                           num_nets=num_nets,
                                                           batch_size=batch_size,
                                                           cross_encoder=cross_encoder,
                                                           langDec_modelpath=langdec_modelpath,
                                                           hypernet=hypernet,
                                                           max_node_size=max_node_size,
                                                           max_edge_size=max_edge_size, checkpoint_epoch=checkpoint_epoch, pos_only=True)
            else:
                #predicted_text, actual_text, some_score
                some_score = 0.0
                predicted_text, actual_text, bleu_score, rouge_score = decoder.demo(langModel=langModel,
                                                                                    archModel=archModel,
                                                                                    langDecModel=langDecModel,
                                                                                    dataset=dataset,
                                                                                    data_path=data_path,
                                                                                    num_nets=num_nets,
                                                                                    batch_size=batch_size,
                                                                                    cross_encoder=cross_encoder,
                                                                                    langDec_modelpath=langdec_modelpath,
                                                                                    hypernet=hypernet,
                                                                                    max_node_size=max_node_size, visualize=visualize,
                                                                                    max_edge_size=max_edge_size, checkpoint_epoch=checkpoint_epoch, pos_only=True)
                if demo:
                    st.markdown('Predicted Text: \r\n' + str(predicted_text))
                    st.markdown('Actual Text: \r\n' + str(actual_text))
                else:
                    print('Predicted Text: \r\n' + str(predicted_text))
                    print('Actual Text: \r\n' + str(actual_text))
            if demo:
                #st.markdown('BLEU Score: %.3f' % (bleu_score))
                st.write(rouge_score)
            else:
                #print("BLEU Score: "+str(bleu_score))
                pprint(rouge_score)
        return bleu_score, rouge_score

if "__main__" == __name__:
    parser = argparse.ArgumentParser(description='ArchBERT test')
    # perform numerical analysis and report accuracy (otherwise, just run the demo)
    parser.add_argument('--validate', dest='validate', action='store_true')
    parser.add_argument('--demo', dest='demo', action='store_true')
    parser.add_argument('--max_node_size', default=512, type=int, help='max_node_size')
    parser.add_argument('--max_edge_size', default=512, type=int, help='max_edge_size')
    # which checkpoint should be used for evaluation. None: last one
    parser.add_argument("--checkpoint_epoch", type=str, default=None, help="Check point epoch")
    # search, reasoning, na_clone_detection, bimodal_na_clone_detection, clone_na_search, bimodal_clone_na_search, qa, langdec
    parser.add_argument('--task', default='reasoning', type=str, help='task type')
    # if task is qa:
    parser.add_argument('--qa_mode', default='multi', type=str, help='qa_mode')
    # 'autonet' or 'tvhf'
    parser.add_argument('--dataset', default='autonet', type=str, help='dataset')
    # the number of archs retrieved as examples using loader
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    # None: collect all data
    parser.add_argument("--num_nets", nargs='?', type=int, const=None, help="num_nets")
    # layernorm works better
    parser.add_argument('--layernorm', dest='layernorm', action='store_true')
    parser.add_argument('--cross_encoder', dest='cross_encoder', action='store_true')
    parser.add_argument('--hypernet', default='gat', type=str, help='hypernet')
    #parser.add_argument('--base_modelname', default='distilbert-base-uncased', type=str, help='base_modelname')
    parser.add_argument('--model_dir', default='archbert_tv', type=str, help='model_dir')
    parser.add_argument('--data_dir', default='cv_final', type=str, help='data_dir')
    parser.add_argument('--visualize', dest='visualize', action='store_true')

    args = parser.parse_args()

    modelpath = args.model_dir
    if args.task == "qa":
        modelpath = modelpath + "_qa" + ("_" + args.qa_mode if args.qa_mode == "multi" else "")
    if args.task == "langdec":
        modelpath = modelpath + "_langdec"
    modelpath = modelpath + ("/"+args.checkpoint_epoch if args.checkpoint_epoch is not None else "")

    if args.dataset == 'tvhf':
        data_path = args.data_dir # vision archs
    elif args.dataset == 'autonet':
        #modelpath = '/home/ma-user/modelarts/user-job-dir/model-search/pretrained-models/archbert/archbert_shape_gat_nodrop_layernorm_deepnet_10k'
        data_path =args.data_dir + ("_qa" if args.task == "qa" else "") #our_data_10k_final/'

    run_task(task=args.task, demo=args.demo, validate=args.validate, batch_size=args.batch_size, dataset=args.dataset,
             num_nets=args.num_nets, roma=None, checkpoint_epoch=args.checkpoint_epoch, hypernet=args.hypernet,
             langModel = None, archModel = None,cross_encoder = args.cross_encoder, max_node_size = args.max_node_size,
             max_edge_size = args.max_edge_size, layernorm=args.layernorm,data_path=data_path,modelpath=modelpath,
             qa_mode=args.qa_mode,visualize=args.visualize)
