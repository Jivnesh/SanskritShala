import React from 'react';
import '../css/resources.css'
function Resources() {
  return <div className='resources'>
    <div className='resourcestitle' >
      <div style={{width:'60%',margin:'auto',textAlign:'left'}} >
        <div style={{fontSize:30,color:'white',padding:10}} >Sanskrit NLP progress</div>
        <div style={{padding:10,color:'white'}} >Repository to track the progress in Sanskrit Natural Language Processing (NLP), including the datasets and the current state-of-the-art for the most common NLP tasks.</div>
      </div>
    </div>
    <div style={{paddingLeft:40}} >
    {/* Word segmentation */}
    <div style={{width:'60%',margin:'auto',textAlign:'left',padding:10}} >
    <h1>Word Segmentation</h1>
    <p>Sanskrit is considered as a cultural heritage and
knowledge preserving language of ancient India.
The momentous development in digitization efforts
has made ancient manuscripts in Sanskrit readily
available for the public domain. However, the usability of these digitized manuscripts is limited due to linguistic challenges posed by the language.
SWS conventionally serves the most fundamental prerequisite for text processing step to make
these digitized manuscripts accessible and to deploy many downstream tasks such as text classification (Sandhan et al., 2019; Krishna et al., 2016b),
morphological tagging (Gupta et al., 2020; Krishna
et al., 2018), dependency parsing (Sandhan et al.,
2021; Krishna et al., 2020a), automatic speech
recognition (Kumar et al., 2022) etc. SWS is not
straightforward due to the phenomenon of sandhi,
which creates phonetic transformations at word
boundaries. This not only obscures the word boundaries but also modifies the characters at juncture
point by deletion, insertion and substitution operation.</p>
    
    <h4>SIGHUM Dataset</h4>
    <p>Currently, Digital Corpus of Sanskrit (Hellwig, 2010, DCS) has more than
600,000 morphologically tagged text lines. It consists of digitized constructions composed in prose
or poetry over a wide span of 3000 years. Summarily, DCS is a perfect representation of various writing styles depending on time and do-
mains. We evaluate on
(Krishna et al., 2017, SIGHUM) dataset. It is a
 subset of DCS (Hellwig, 2010). This dataset
also come with candidate solution space generated
by SHR for SWS. We prefer Krishna et al. (2017,
SIGHUM) over a relatively larger dataset (Hellwig and Nehrdich, 2018) to obviate the time and
efforts required for obtaining candidate solution
space. We obtain the ground truth segmentation solutions from DCS. We could not use DCS10k (Krishna et al., 2020b) due to partly missing gold
standard segmentation (inflections) for almost 50%
data points. SIGHUM consists of 97,000, 3,000
and 4,200 sentences as train, dev, test set, respectively. We use the following word-level
evaluation metrics: macro-averaged Precision (P),
Recall (R), F1-score (F) and the percentage of sentences with perfect matching (PM)</p>
    <table>
      <tr>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:40}}>Model</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>Paper</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>Code</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>P</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>R</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>F</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>PM</th>
      </tr>

    <tr>
        <td>Seq2seq</td>
        <td><a href="https://aclanthology.org/L18-1264/">Building a Word Segmenter for Sanskrit Overnight</a></td>
        <td><a href="https://github.com/cvikasreddy/skt">Official</a></td>
        <td>73.44</td>
        <td>73.04</td>
        <td>73.24</td>
        <td>29.20</td>
    </tr>
    <tr>
        <td>SupPCRW</td>
        <td><a href="https://aclanthology.org/C16-1048.pdf">Word segmentation in sanskrit using path constrained random walks</a></td>
        <td><a href="">NA</a></td>
        <td>76.30</td>
        <td>79.47</td>
        <td>77.85</td>
        <td>38.64</td>
    </tr>
    <tr>
        <td>AttsegSeq2seq</td>
        <td><a href="https://aclanthology.org/L18-1264/">Building a Word Segmenter for Sanskrit Overnight</a></td>
        <td><a href="https://github.com/cvikasreddy/skt">Official</a></td>
        <td>90.77</td>
        <td>90.30</td>
        <td>90.53</td>
        <td>55.99</td>
    </tr>
    <tr>
        <td>TENER</td>
        <td><a href="https://arxiv.org/abs/1911.04474">TENER: Adapting Transformer Encoder for Named Entity Recognition</a></td>
        <td><a href="https://github.com/fastnlp/TENER">Official</a></td>
        <td>90.03</td>
        <td>89.20</td>
        <td>89.61</td>
        <td>61.24</td>
    </tr>
    <tr>
        <td>Lattice-LSTM</td>
        <td><a href="https://aclanthology.org/P18-1144/">Chinese NER Using Lattice LSTM</a></td>
        <td><a href="https://github.com/jiesutd/LatticeLSTM">Official</a></td>
        <td>94.36</td>
        <td>93.83</td>
        <td>94.09</td>
        <td>76.99</td>
    </tr>
    <tr>
        <td>Lattice-GNN</td>
        <td><a href="https://aclanthology.org/D19-1096/">A Lexicon-Based Graph Neural Network for Chinese NER</a></td>
        <td><a href="https://github.com/RowitZou/LGN">Official</a></td>
        <td>95.76</td>
        <td>95.24</td>
        <td>95.50</td>
        <td>81.58</td>
    </tr>
    <tr>
        <td>Transformer</td>
        <td><a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</a></td>
        <td><a href="https://github.com/facebookresearch/fairseq">Official</a></td>
        <td>96.52</td>
        <td>96.21</td>
        <td>96.36</td>
        <td>83.88</td>
    </tr>
    <tr>
        <td>FLAT-Lattice</td>
        <td><a href="https://aclanthology.org/2020.acl-main.611/">FLAT: Chinese NER Using Flat-Lattice Transformer</a></td>
        <td><a href="https://github.com/LeeSureman/Flat-Lattice-Transformer">Official</a></td>
        <td>96.75</td>
        <td>96.70</td>
        <td>96.72</td>
        <td>85.65</td>
    </tr>
    <tr>
        <td>Cliq-EBM</td>
        <td><a href="https://aclanthology.org/D18-1276/">Free as in Free Word Order: An Energy Based Model for Word Segmentation and Morphological Tagging in Sanskrit</a></td>
        <td><a href="https://zenodo.org/record/1035413#.W35s8hjhUUs">Official</a></td>
        <td>96.18</td>
        <td>97.67</td>
        <td>96.92</td>
        <td>78.83</td>
    </tr>
    <tr>
        <td>rcNN-SS</td>
        <td><a href="https://aclanthology.org/D18-1295/">Sanskrit Word Segmentation Using Character-level Recurrent and Convolutional Neural Networks</a></td>
        <td><a href="https://github.com/OliverHellwig/sanskrit/tree/master/papers/2018emnlp">Official</a></td>
        <td>96.86</td>
        <td>96.83</td>
        <td>96.84</td>
        <td>87.08</td>
    </tr>
    <tr>
        <td>TransLISTngram</td>
        <td><a href="https://arxiv.org/abs/2210.11753">TransLIST: A Transformer-Based Linguistically Informed Sanskrit Tokenizer</a></td>
        <td><a href="https://github.com/rsingha108/TransLIST">Official</a></td>
        <td>96.97</td>
        <td>96.77</td>
        <td>96.87</td>
        <td>86.52</td>
    </tr>
    <tr>
        <td>TransLIST</td>
        <td><a href="https://arxiv.org/abs/2210.11753">TransLIST: A Transformer-Based Linguistically Informed Sanskrit Tokenizer</a></td>
        <td><a href="https://github.com/rsingha108/TransLIST">Official</a></td>
        <td>98.80</td>
        <td>98.93</td>
        <td>98.86</td>
        <td>93.97</td>
    </tr>
</table>
  

    <h4>Hackathon Dataset</h4>
    <p>We also evaluate on (Krishnan
et al., 2020, Hackathon) for SWS. This dataset
is a subset of DCS (Hellwig, 2010). It
also comes with candidate solution space generated
by SHR for SWS. Hackathon consists of 90,000,
10,332 and 9,963 sentences as train, dev and test
set, respectively. We use the following word-level
evaluation metrics: macro-averaged Precision (P),
Recall (R), F1-score (F) and the percentage of sentences with perfect matching (PM)</p>
<table>
      <tr>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:40}}>Model</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>Paper</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>Code</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>P</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>R</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>F</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>PM</th>
      </tr>

    <tr>
        <td>Seq2seq</td>
        <td><a href="https://aclanthology.org/L18-1264/">Building a Word Segmenter for Sanskrit Overnight</a></td>
        <td><a href="https://github.com/cvikasreddy/skt">Official</a></td>
        <td>72.31</td>
        <td>72.15</td>
        <td>72.23</td>
        <td>20.21</td>
    </tr>
    <tr>
        <td>TENER</td>
        <td><a href="https://arxiv.org/abs/1911.04474">TENER: Adapting Transformer Encoder for Named Entity Recognition</a></td>
        <td><a href="https://github.com/fastnlp/TENER">Official</a></td>
        <td>89.38</td>
        <td>87.33</td>
        <td>88.35</td>
        <td>49.92</td>
    </tr>
    <tr>
        <td>Lattice-LSTM</td>
        <td><a href="https://aclanthology.org/P18-1144/">Chinese NER Using Lattice LSTM</a></td>
        <td><a href="https://github.com/jiesutd/LatticeLSTM">Official</a></td>
        <td>91.47</td>
        <td>89.19</td>
        <td>90.31</td>
        <td>65.76</td>
    </tr>
    <tr>
        <td>Lattice-GNN</td>
        <td><a href="https://aclanthology.org/D19-1096/">A Lexicon-Based Graph Neural Network for Chinese NER</a></td>
        <td><a href="https://github.com/RowitZou/LGN">Official</a></td>
        <td>92.89</td>
        <td>94.31</td>
        <td>93.59</td>
        <td>70.31</td>
    </tr>
    <tr>
        <td>Transformer</td>
        <td><a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</a></td>
        <td><a href="https://github.com/facebookresearch/fairseq">Official</a></td>
        <td>95.79</td>
        <td>95.23</td>
        <td>95.51</td>
        <td>77.7</td>
    </tr>
    <tr>
        <td>FLAT-Lattice</td>
        <td><a href="https://aclanthology.org/2020.acl-main.611/">FLAT: Chinese NER Using Flat-Lattice Transformer</a></td>
        <td><a href="https://github.com/LeeSureman/Flat-Lattice-Transformer">Official</a></td>
        <td>96.44</td>
        <td>95.43</td>
        <td>95.93</td>
        <td>77.94</td>
    </tr>
    <tr>
        <td>rcNN-SS</td>
        <td><a href="https://aclanthology.org/D18-1295/">Sanskrit Word Segmentation Using Character-level Recurrent and Convolutional Neural Networks</a></td>
        <td><a href="https://github.com/OliverHellwig/sanskrit/tree/master/papers/2018emnlp">Official</a></td>
        <td>96.4</td>
        <td>95.15</td>
        <td>95.77</td>
        <td>77.62</td>
    </tr>
    <tr>
        <td>TransLISTngram</td>
        <td><a href="https://arxiv.org/abs/2210.11753">TransLIST: A Transformer-Based Linguistically Informed Sanskrit Tokenizer</a></td>
        <td><a href="https://github.com/rsingha108/TransLIST">Official</a></td>
        <td>96.68</td>
        <td>95.74</td>
        <td>96.21</td>
        <td>79.28</td>
    </tr>
    <tr>
        <td>TransLIST</td>
        <td><a href="https://arxiv.org/abs/2210.11753">TransLIST: A Transformer-Based Linguistically Informed Sanskrit Tokenizer</a></td>
        <td><a href="https://github.com/rsingha108/TransLIST">Official</a></td>
        <td>97.78</td>
        <td>97.44</td>
        <td>97.61</td>
        <td>85.47</td>
    </tr>
</table>
    </div>
    {/* Dependency parsing */}
    <div style={{width:'60%',margin:'auto',textAlign:'left',padding:10}} >
    <h1>Dependency parsing</h1>
    <p>Dependency parsing is the task of extracting a dependency parse of a sentence that represents its grammatical structure and defines the relationships between “head” words and words, which modify those heads.
    Example: Relations among the words are illustrated above the sentence with directed, labeled arcs from heads to dependents (+ indicates the dependent).
    </p>
    
    <h4>Sanskrit Tree Bank Corpus</h4>
    <p>The dataset for the dependency parsing is obtained from the department of Sanskrit studies, UoHyd.
       We use about 1,700 prose sentences from the Sanskrit Tree Bank Corpus, henceforth to be referred
        to as STBC (Kulkarni et al., 2010; Kulkarni, 2013). Further, we use 1,300 sentences (including 300 prose datapoints from Shishupalavadha) from 
        STBC as the test set and the remaining 1000 as a dev set.  
        The final results on the test set are reported using systems trained with combined gold train and dev set.</p>
    <table>
      <tr>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:40}}>Model</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>UAS</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>LAS</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>Paper/Source</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>Code</th>
      </tr>
    <tr>
      <td>YAP</td>
      <td>75.31</td>
      <td>66.02</td>
      <td><a href="https://www.aclweb.org/anthology/Q19-1003">Joint Transition-Based Models for Morpho-Syntactic Parsing: Parsing Strategies for MRL's and a Case Study from Modern Hebrew</a></td>
      <td><a href="https://github.com/OnlpLab/yap">Official</a></td>
    </tr>
    <tr>
      <td>L2S</td>
      <td>81.97</td>
      <td>74.14</td>
      <td><a href="http://users.umiacs.umd.edu/~hal/docs/daume16compiler.pdf">A Credit Assignment Compiler for Joint Prediction</a></td>
      <td><a href="https://github.com/OnlpLab/yap">Official</a></td>
    </tr>
    <tr>
      <td>Tree-EBM-F</td>
      <td>82.65</td>
      <td>79.28</td>
      <td><a href="https://aclanthology.org/D18-1276/">Free as in Free Word Order: An Energy Based Model for Word Segmentation and Morphological Tagging in Sanskrit</a></td>
      <td><a href="https://zenodo.org/record/1035413#.W35s8hjhUUs">Official</a></td>
    </tr>
    <tr>
      <td>BiAFF</td>
      <td>85.88</td>
      <td>79.55</td>
      <td><a href="https://arxiv.org/abs/1611.01734">Deep Biaffine Attention for Neural Dependency Parsing</a></td>
      <td><a href="https://github.com/rotmanguy/DCST">Official</a></td>
    </tr>
    <tr>
      <td>Tree-EBM-F*</td>
      <td>85.88</td>
      <td>79.55</td>
      <td><a href="https://aclanthology.org/D18-1276/">Free as in Free Word Order: An Energy Based Model for Word Segmentation and Morphological Tagging in Sanskrit</a></td>
      <td><a href="https://zenodo.org/record/1035413#.W35s8hjhUUs">Official</a></td>
    </tr>
    <tr>
      <td>MG-EBM*</td>
      <td>85.88</td>
      <td>79.55</td>
      <td><a href="https://aclanthology.org/2020.emnlp-main.388/">Keep it Surprisingly Simple: A Simple First Order Graph Based Parsing Model for Joint Morphosyntactic Parsing in Sanskrit</a></td>
      <td><a href="">NA</a></td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>88.67</td>
      <td>83.47</td>
      <td><a href="https://arxiv.org/pdf/2201.11374.pdf">Systematic Investigation of Strategies Tailored for Low-Resource Settings for
Sanskrit Dependency Parsing</a></td>
      <td><a href="https://github.com/Jivnesh/sandp">Official</a></td>
    </tr>
  
    </table>  
      
    <h4>Universal Dependencies: Vedic Sanskrit Treebank</h4>
    <p> We also evaluate on the Vedic Sanskrit Treebank (Hellwig et al., 2020, VST) consisting of 1,500 , 1,024 and 1,473 sentences (poetry-prose mixed) as train, dev and test data, respectively. For both data, the final results on the test set are reported using systems trained with combined gold train and dev set.</p>
    <table>
    <tr>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:40}} >Model</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>UAS</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>LAS</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>Paper/Source</th>
        <th style={{backgroundColor:'rgb(77, 72, 72)',color:'white',padding:10}}>Code</th>
      </tr>
      <tr>
      <td>YAP</td>
      <td>70.37</td>
      <td>56.09</td>
      <td><a href="https://www.aclweb.org/anthology/Q19-1003">Joint Transition-Based Models for Morpho-Syntactic Parsing: Parsing Strategies for MRL's and a Case Study from Modern Hebrew</a></td>
      <td><a href="https://github.com/OnlpLab/yap">Official</a></td>
    </tr>
    <tr>
      <td>L2S</td>
      <td>72.44</td>
      <td>62.76</td>
      <td><a href="http://users.umiacs.umd.edu/~hal/docs/daume16compiler.pdf">A Credit Assignment Compiler for Joint Prediction</a></td>
      <td><a href="https://github.com/OnlpLab/yap">Official</a></td>
    </tr>
    <tr>
      <td>BiAFF</td>
      <td>77.23</td>
      <td>67.68</td>
      <td><a href="https://arxiv.org/abs/1611.01734">Deep Biaffine Attention for Neural Dependency Parsing</a></td>
      <td><a href="https://github.com/rotmanguy/DCST">Official</a></td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>79.71</td>
      <td>69.89</td>
      <td><a href="https://arxiv.org/pdf/2201.11374.pdf">Systematic Investigation of Strategies Tailored for Low-Resource Settings for
Sanskrit Dependency Parsing</a></td>
      <td><a href="https://github.com/Jivnesh/sandp">Official</a></td>
    </tr>
</table>
    </div>
    <p><a href="http://cnerg.iitkgp.ac.in/sanskritshala/">Go back to the Home</a></p>
    </div>
  </div>;
}

export default Resources;
