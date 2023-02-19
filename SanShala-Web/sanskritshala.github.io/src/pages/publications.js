import React from 'react';
import {FaFile,FaReadme,FaPagelines,FaTeamspeak,FaHome} from 'react-icons/fa'
import Sidebar from '../custom/sidebar';
import Title from '../custom/title'
import '../css/publications.css'
function publications({width}) {
  return <div>
      <Title width={width} page={"Publications"} />
      <div style={{padding:20,textAlign:'left'}} >
          <div style={{textAlign:'center',fontSize:'larger',padding:10,fontWeight:'bolder',
        }} >
          Publications
          </div>
          <div className='publicationstitle' >
          Conferences
          </div>
    
          <li><span style={{fontWeight:'bold'}} >Jivnesh Sandhan</span>, Laxmidhar Behera, Pawan Goyal. Systematic Investigation of Strategies
Tailored for Low-Resource Settings for Low-Resource Dependency Parsing. Proceedings
of the European Chapter of the Association for Computational Linguistics, EACL 2023,
Dubrovnik, Croatia.</li>
          <li><span style={{fontWeight:'bold'}} >Jivnesh Sandhan</span>, Rathin Singha, Narein Rao, Suvendu Samanta, Laxmidhar Behera,
Pawan Goyal. TransLIST: A Transformer-Based Linguistically Informed Sanskrit Tokenizer.
Proceedings of the Conference on Empirical Methods in Natural Language Processing,
EMNLP (Findings) 2022, Abu Dhabi</li>
          <li><span style={{fontWeight:'bold'}} >Jivnesh Sandhan</span>, Ashish Gupta, Hrishikesh Terdalkar, Tushar Sandhan, Suvendu Samanta,
Laxmidhar Behera and Pawan Goyal. A Novel Multi-Task Learning Approach for ContextSensitive Compound Type Identification in Sanskrit. Proceedings of International Conference on Computational Linguistics, COLING 2022, Republic of Korea</li>
          <li><span style={{fontWeight:'bold'}} >Jivnesh Sandhan</span>, Om Adideva, Digumarthi Komal, Laxmidhar Behera, Pawan Goyal. Evaluating Neural Word Embeddings for Sanskrit. In Proceedings of the World Sanskrit Conference, WSC 2023, Canberra, Australia.</li>
          <li>Amrith Krishna, Ashim Gupta, Deepak Garasangi, <span style={{fontWeight:'bold'}} >Jivnesh Sandhan</span>, Pavankumar Satuluri, Pawan Goyal. Neural Approaches for Data-Driven Dependency Parsing in Sanskrit. World Sanskrit Conference, WSC 2023, Canberra, Australia</li>
          <li><span style={{fontWeight:'bold'}} >Jivnesh Sandhan</span>, Amrith Krishna, Pawan Goyal, and Laxmidhar Behera. Revisiting the
role of feature engineering for compound type identification in Sanskrit. In Proceedings of
the 6th International Sanskrit Computational Linguistics Symposium, ISCLS 2019, Kharagpur, India.</li>
           <div className='publicationstitle' >
          Workshops
          </div>
          <li><span style={{fontWeight:'bold'}} >Jivnesh Sandhan</span>, Amrith Krishna, Ashim, Gupta, Pawan Goyal, and Laxmidhar Behera.
A Little Pretraining Goes a Long Way: A Case Study on Dependency Parsing Task for Lowresource Morphologically Rich Languages. Proceedings of the European Chapter of the
Association for Computational Linguistics, EACL-SRW 2021, Ukraine</li>
<li><span style={{fontWeight:'bold'}} >Jivnesh Sandhan</span>, Ayush Daksh, Om Adideva Paranjay, Laxmidhar Behera, Pawan Goyal.
Prabhupadavani: A Code-mixed Speech Translation Data for 26 Languages. Proceedings of International Conference on Computational Linguistics Workshop on Computational
Linguistics for Cultural Heritage, Social Sciences, Humanities, and Literature, COLINGLaTeCH-CLfL 2022, Gyeongju, Republic of Korea</li>
          
          <div className='publicationstitle' >
          Preprints
          </div>
          <li><span style={{fontWeight:'bold'}} >Jivnesh Sandhan</span>, Laxmidhar Behera, Pawan Goyal 2022. Systematic Investigation of Strategies Tailored for Low-Resource Settings for Sanskrit Dependency Parsing.</li>
          
          
      </div>

  </div>;
}

export default publications;
