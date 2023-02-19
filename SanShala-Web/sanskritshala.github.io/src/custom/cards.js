import React,{useState} from 'react';
import '../css/cards.css'
import $ from 'jquery'
import '@fontsource/roboto'
function Cards({vid,title,vcolor1,vcolor2,...props}) {
    var vvid="#"+vid;
    const [width,setwidth]=useState(window.innerWidth)
    window.addEventListener('change',()=>{
      setwidth(window.innerWidth)
    })
    vcolor2=vcolor1
     vcolor1=(vcolor1.substr(0,vcolor1.indexOf(')')))+",0.25)"
     vcolor2=(vcolor2.substr(0,vcolor2.indexOf(')')))+",0.01)"
     
  return <div className='cards' {...props} >
      <div className='tile' id={vid} style={{backgroundImage:'linear-gradient(90deg,'+vcolor1+','+vcolor2+')',
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
        Please click here to get started with this task. This application is integrated with our pretrained state-of-the-art model. You may use it for creating annotated dataset.
        </div>
      </div>
  </div>;
}

export default Cards;
