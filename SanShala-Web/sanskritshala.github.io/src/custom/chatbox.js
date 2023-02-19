import React, { useState } from 'react'
import '../css/chatbox.css'
import $ from 'jquery';
import { FaFacebookMessenger } from 'react-icons/fa';
import chatbot from '../images/chatbot.png'
import TypeAnimation from 'react-type-animation'
function Chatbox({setval,val,cc,setcc}) {
    $('.chatboxchatarea').css({'height':val?'90%':''})
    var count=0;
    var jj=[]
    const [input,setinput]=useState('')
    var responses={
        "Can you please let me know how can I help you?":["Give me a demo",'Motivation behind SanskritShala','Applications of SanskritShala', 'More about modules'],
        "Motivation behind SanskritShala":['To improve the accessibily of Sanskrit manuscript','Applications of SanskritShala','Main menu',"Thanks"],
        "To improve the accessibily of Sanskrit manuscript":['Main menu','Thanks'],
        "Thanks":["Thanks for visiting our platform.","We appreciate your feedback"],
        "Applications of SanskritShala":['It can serve a annotation tooklit','Eventually can be used for pedalogical purposes','More about modules','Main menu'],
        "It can serve a annotation tooklit":['Main menu', "Thanks"],
        "Eventually can be used for pedalogical purposes":['Main menu', "Thanks"],
        "Give me a demo":["Coming soon",'Main menu', "Thanks"],
        "Coming soon":['Main menu', "Thanks"],
        "Main menu":["Give me a demo",'Motivation behind SanskritShala','Applications of SanskritShala', 'More about modules'],
        "More about modules":['Word Segmentation','Morphological Tagger','Dependency Parser','Compound Idetifier'],
        "Word Segmentation":['Click on its app to use it',"Paper:",'Code:','Main menu','Thanks'],
        "Morphological Tagger":['Click on its app to use it',"Paper",'Code:','Main menu','Thanks'],
        "Dependency Parser":['Click on its app to use it',"Paper:",'Code:','Main menu','Thanks'],
        "Compound Idetifier":['Click on its app to use it',"Paper:",'Code:','Main menu','Thanks'],
        "Click on its app to use it":['Main menu', "Thanks",'We appreciate your feedback'],
        "Thanks for visiting our platform.":['Please write us at: jivneshsandhan@gmail.com'],
        "We appreciate your feedback":['Please write us at: jivneshsandhan@gmail.com','Done'],
        "Please write us at: jivneshsandhan@gmail.com":['Done'],
        "Done":['Have a nice day!']
    }
    cc.map(values=>(
        jj.push(values)
    ))
    // $('.chatinput').on('keyup', (key)=>{
    //     if(key.which==13){
    //         var input=document.getElementsByClassName('chatinput')[0].value ;
    //         console.log(input)
    //         jj.reverse()
    //         jj.push({user:input})
    //         setcc(jj)
    //     }
    // })
    jj.reverse();
    return (
        <div className='chatbox' style={{zIndex:1111,position:'fixed'}} >
            <div className='chatboxchatarea'>
                <div className='chatarea' >
                {responses[jj[0].bot]&& <div style={{
                backgroundColor:'whitesmoke',
                }} >
                    {
                         responses[jj[0].bot].map(values=>(
                            <div className='chatresponses'
                            onClick={()=>{
                                jj.reverse();
                                jj.push({user:values})
                                jj.push({bot:values})
                                setcc(jj)
                            }}
                            >
                                {values}
                                </div>
                        ))
                    }
                </div>}
                    {
                        
                        jj.map(values=>(
                            <div style={{ backgroundColor:'whitesmoke',display:'flex',justifyContent:'space-between',
                            paddingTop:10,
                            }} >

                                {values.bot&&<div style={{width:'100%',display:'flex',justifyContent:'left'}} >
                                    <p style={{
                                marginLeft:4,
                                backgroundColor:'rgba(145, 8, 255, 0.56)',
                                paddingLeft:12,
                                paddingRight:12,
                                paddingTop:12,
                                borderTopRightRadius:12,
                                borderBottomRightRadius:12,
                                borderTopLeftRadius:4,
                                borderBottomLeftRadius:4,
                                maxWidth:'75%',
                                color:'white',
                                fontSize:12,
                                textAlign:'left'
                            
                            }} >{values.bot===jj[0].bot? <TypeAnimation sequence={[values.bot,1000]} />:<p>{values.bot} </p> } </p> 
                            </div>
                            }
                                {values.user&&
                                <div style={{width:'100%',display:'flex',justifyContent:'right'}} >
                                <div style={{
                                    marginRight:4,
                                    backgroundColor:'rgba(100, 109, 118, 0.733)',
                                    paddingRight:12,
                                    paddingLeft:12,
                                    borderTopLeftRadius:12,
                                    borderBottomLeftRadius:12,
                                    borderTopRightRadius:4,
                                    borderBottomRightRadius:4,
                                    maxWidth:'75%',
                                    color:'white',
                                    textAlign:'left',
                                    fontSize:12,
                                
                                
                            
                            }} ><p>{values.user} </p> </div>
                            </div>
                            }
                            </div>
                        ))
                    }
                    
                </div>
                <div style={{
                backgroundColor:'whitesmoke',
                maxWidth:'100%',
                paddingTop:'2.5%',
                height:'15%',
                display:'flex',
                flexDirection:'row',
                overflow:'auto',
                marginLeft:4,
                marginRight:4,
            }} >
                <input className='chatinput' style={{
                    height:'70%',
                    width:'100%',
                    borderRadius:40,
                    marginLeft:10,
                    marginRight:10,
                    paddingLeft:20,
                    backgroundColor:'rgba(248, 249, 250, 0.733)',
                    borderStyle:'solid',
                    
                    
                }}
                id="chatinput"
                value={input}
                onChange={()=>{
                    
                    setinput(document.getElementById("chatinput").value.toString())
                    $('#chatinput').keypress((key)=>{
                        if(key.which===13&&document.getElementById('chatinput').value.toString()>""){
                            jj.reverse()
                            jj.push({'user':document.getElementById("chatinput").value.toString()})
                            setcc(jj)
                            setinput("")
                        }
                    })
                }}
                />
                </div>
            
        </div>
            {val&&
                <div className='titlechatbox' 
               onClick={()=>{
                   $('.chatboxchatarea').css({'height':++count&1?'0%':''})
                
            }}
            
            >

                <img src={chatbot} style={{borderRadius:'50%',marginTop:4,marginBottom:4,marginLeft:4}} />
                <span style={{color:'white',alignSelf:'center',paddingLeft:12}} >
                    SanskritShala Bot
                    </span>
                <FaFacebookMessenger color='white' className='chattitleicon' style={{alignSelf:'center',
            transition:'color .4s'    ,paddingRight:12
            }}
                onMouseEnter={()=>{
                    $('.chattitleicon').css({'color':'red'})
                }}
                onMouseLeave={()=>{
                    $('.chattitleicon').css({'color':'white'})
                }}
                onClick={()=>{
                    $('.chatbox').css({});
                    setval(1)
                }} />
            </div>}
            
        </div>
    )
}

export default React.memo(Chatbox)