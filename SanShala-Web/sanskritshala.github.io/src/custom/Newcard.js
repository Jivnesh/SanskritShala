import React,{useState} from 'react';
import '../css/cards.css'
import $ from 'jquery'
import '@fontsource/roboto'
function NewCards({vid,title,vcolor1,vcolor2,...props}) {
    var vvid="#"+vid;
    const [width,setwidth]=useState(window.innerWidth)
    window.addEventListener('change',()=>{
      setwidth(window.innerWidth)
    })
    vcolor2=vcolor1
     vcolor1=(vcolor1.substr(0,vcolor1.indexOf(')')))+",0.25)"
     vcolor2=(vcolor2.substr(0,vcolor2.indexOf(')')))+",0.01)"
     
  return <div className='cards' {...props} >
      <div className='newtile' id={vid} style={{backgroundImage:'linear-gradient(90deg,'+vcolor1+','+vcolor2+')',
    borderColor:vcolor1
    }} 
      >
        <div style={{
      }} >
        <div className='tiletitle' >
            {title}
          </div>
          
        </div>
        <div className='description' >
        SanskritShala (translation: school of Sansksrit) is a neural Sanskrit Natural Language Processing (NLP) toolset designed to allow computational linguistic analysis for a variety of tasks, including word segmentation, morphological tagging, dependency parsing, and compound type recognition. Our systems perform at the cutting edge on all accessible benchmark datasets for all tasks. SanskritShala is a web-based toolkit that provides the user with real-time analysis for the input provided. It is designed with user-friendly interactive data annotation tools that allow annotators to amend the system's incorrect predictions. We share the source codes of the four modules contained in the toolkit, six word embedding models trained on publicly accessible Sanskrit corpora, and various annotated datasets for evaluating the intrinsic features of word embeddings. According to our knowledge, this is the first neural-based Sanskrit NLP toolbox with a web interface and many NLP modules. We are certain that practitioners of Sanskrit computational linguistics (SCL) will find it beneficial for instructional and annotation purposes. Our platform's video demonstration is coming soon.
        </div>
      </div>
  </div>;
}

export default NewCards;
