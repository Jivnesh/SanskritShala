import React, { useState } from 'react'
import '../css/Old_Dp.css'
import $ from 'jquery'
import { Helmet } from 'react-helmet';
import Xarrow, { Xwrapper } from 'react-xarrows'
import {FaHamburger,FaArrowRight,FaArrowLeft,FaArrowUp,FaArrowDown ,FaUpload} from 'react-icons/fa'
import Menu from '../custom/Menu';
function OldDp() {
    const [input,setinput]=useState("");
    const [display,setdisplay]=useState({});
    const [data,setdata]=useState({});
    const [tag,settag]=useState('prayojanam')
    const [filedata,setfiledata]=useState()
    const [dataval,setdataval]=useState(1)
    const [toggle,settoggle]=useState(0)
    
    const [searchtag,setsearchtag]=useState('')
    console.log(display)
    let numval=18*Math.random()
    let color={'samucciwam': 'rgb(151,151,151)', 'axikaranam': 'rgb(40.892404046057756,195.90212002826695,248.29351969076725)', 'karmasamanaxikaranam': 'rgb(80.76486511906698,52.51440400293031,53.08090307846933)', 'karma': 'rgb(88.5884063088804,203.67910423153953,209.90796543767718)', 'sampraxanam': 'rgb(129.61976034829996,229.08347665891358,173.3997126553231)', 'apaxanam': 'rgb(228.43215159482878,227.79103939439756,164.96817855566414)', 'bavalakranasapwami_samanakalah': 'rgb(62.66904930870843,129.16280789049665,87.1932825655106)', 'karanam': 'rgb(40.934260339109095,211.0082341777622,43.83889586912186)', 'prawiyogi': 'rgb(137.1310137798212,16.568779621860443,74.52970531571357)', 'anuyogi': 'rgb(63.80087183168032,157.6029046976205,78.49846088699618)', 'root': 'rgb(110.55401377865476,44.29834990456047,198.69765003909146)', 'karwqsamanaxikaranam': 'rgb(25.7895756160927,32.093979258717745,26.327904868555837)', 'rartisambanxah': 'rgb(144.3227317950415,144.5790378977648,231.3117286758576)', 'purvakalah': 'rgb(95.2544330750636,184.51382417829535,72.09252948336751)', 'upapaxasambanxah': 'rgb(69.44475016890506,133.2935468459755,83.99722071535241)', 'waxarwyam': 'rgb(61.463645799040926,26.082897978324898,162.25828698139338)', 'hewuh': 'rgb(6.673138254210781,226.72974767628264,91.26865517995041)', 'kriyaviseranam': 'rgb(176.11552945879745,240.75046624369853,234.98471431815648)', 'viseranam': 'rgb(250.28726103194137,103.98357591697282,13.88006310154271)', 'sambanxah': 'rgb(129.19527775529997,158.05623007107332,40.577465250224115)', 'samboxyah': 'rgb(70.12094532173067,152.7382702021216,209.3796755632114)', 'prayojanam': 'rgb(15.663448901199954,189.05485993705983,116.69688888451144)', 'karwa': 'rgb(37.632586745822074,74.82709935392526,79.74058327798153)'}
    var num=1,k1=-1,k2=-1;
    const[tags,settags]=useState({'prayojanam': 1, 'sampraxanam': 0, 'purvakalah': 0, 'samucciwam': 0, 'anuyogi': 0, 'karwqsamanaxikaranam': 0, 'bavalakranasapwami_samanakalah': 0, 'hewuh': 0, 'apaxanam': 0, 'root': 0, 'samboxyah': 0, 'upapaxasambanxah': 0, 'karma': 0, 'karmasamanaxikaranam': 0, 'axikaranam': 0, 'sambanxah': 0, 'waxarwyam': 0, 'kriyaviseranam': 0, 'prawiyogi': 0, 'karanam': 0, 'karwa': 0, 'viseranam': 0, 'rartisambanxah': 0}    
        )
        let valcolor=['red','green','pink','yellow']
  return (
      <div className='dp'  >

          <Helmet>
          <script src='leader-line.min.js' ></script>
          
          </Helmet>
          <div className='dpdisplay' >
              <div>
                  <div style={{color:'black',display:'flex'}}>
                      
            <textarea onChange={()=>{
                setinput(document.getElementById('dpinput').value)
            }}  id='dpinput' className='dpinput' placeholder='Text...'/>
              
          <button className='submit' onClick={()=>{
              var jj={},jl=input.split(' '),jdata={};
              for(let j=0;j<jl.length;j++){
                jj[j+1] ={w1:jl[j],color:[],w:''}
                jdata[j+1]={w1:jl[j],color:[]}
                //   jj[j+1].color=' ';
                //   jj[j+1].w=''
              }
              setdata(jdata)
              setdisplay(jj)
              Object.keys(data).map(val=>(
                  $('#'+val).css({backgroundColor:'',color:'black'})
              ))
          }} >
              Submit
          </button>
          <input type={'file'} id='fileupload' className='fileupload' name='file' onChange={()=>{
                var fileToLoad = document.getElementById("fileupload").files[0];

  var fileReader = new FileReader();
  fileReader.onload = function(fileLoadedEvent){
      var jj = fileLoadedEvent.target.result;
      jj=JSON.parse(jj)
      setfiledata(jj)
      
      setdata(filedata[dataval])
      setdisplay(filedata[dataval])
    };

  fileReader.readAsText(fileToLoad, "UTF-8");
                // setfiledata(document.getElementById('fileinput').value)
            }} />
          <label className='upload' for='fileupload'  >
              <FaUpload className='searchicon' color='blue' />
          </label>
              </div>
              <div style={{color:'white',display:'flex',backgroundColor:'blue'}} >
              <div
              
              
              style={{width:'85%', color:'white',justifyContent:'space-evenly',backgroundColor:'blue',padding:12,overflowX:'auto'}} >
              {
                  Object.keys(tags).map(val=>(
                      <span className='dpwords' style={{backgroundColor:tags[val]?color[tag] :'',color:'white',borderColor:'white',
                
                    }}
                      onClick={()=>{
var tagval=                        {'prayojanam': 0, 'sampraxanam': 0, 'purvakalah': 0, 'samucciwam': 0, 'anuyogi': 0, 'karwqsamanaxikaranam': 0, 'bavalakranasapwami_samanakalah': 0, 'hewuh': 0, 'apaxanam': 0, 'root': 0, 'samboxyah': 0, 'upapaxasambanxah': 0, 'karma': 0, 'karmasamanaxikaranam': 0, 'axikaranam': 0, 'sambanxah': 0, 'waxarwyam': 0, 'kriyaviseranam': 0, 'prawiyogi': 0, 'karanam': 0, 'karwa': 0, 'viseranam': 0, 'rartisambanxah': 0}   

                          tagval[val]=1;
                          settags(tagval)
                          settag(val)
                      }}
                      >
                          {val}
                          </span>
                  
                  ))
              }
              </div>
              
              <div style={{width:'15%',borderLeftStyle:'solid'}}

              onClick={()=>{
                  settoggle(!toggle)
              }}
              >
                  {
                      toggle==1?<FaArrowUp className='searchicon' />:<FaArrowDown className='searchicon' />
                  }
              </div>
              </div>
              </div>
              </div>
             
              <div className='display' >
                  <div style={{color:'black',position:'absolute',top:0,padding:2}} >
Tag selected: <span style={{color:'black',backgroundColor:color[tag],borderRadius:4,padding:2}} >{tag} </span>
                  </div>
              {toggle==1&& <div className='searchwindow' style={{}} >
                  <input id='searchtag' onChange={()=>{
                      var val=document.getElementById('searchtag').value;
                      setsearchtag(val);
                  }} />
                  <div style={{
                      
                      color:'black',maxHeight:'85%',overflow:'auto'}} >
                      {
                          Object.keys(tags).map(val=>(
                              val.startsWith(searchtag)?
                              <div className='searchoptions' 
                              onClick={()=>{
                                var tagval=                        {'prayojanam': 0, 'sampraxanam': 0, 'purvakalah': 0, 'samucciwam': 0, 'anuyogi': 0, 'karwqsamanaxikaranam': 0, 'bavalakranasapwami_samanakalah': 0, 'hewuh': 0, 'apaxanam': 0, 'root': 0, 'samboxyah': 0, 'upapaxasambanxah': 0, 'karma': 0, 'karmasamanaxikaranam': 0, 'axikaranam': 0, 'sambanxah': 0, 'waxarwyam': 0, 'kriyaviseranam': 0, 'prawiyogi': 0, 'karanam': 0, 'karwa': 0, 'viseranam': 0, 'rartisambanxah': 0}   

                          tagval[val]=1;
                          settags(tagval)
                          settoggle(0)
                          settag(val)
     
                              }}
                              
                              >
                                  {val}
                                  </div>:''
                          ))
                      }
                  </div>
              </div>
             }
              <div >
              {
                  Object.keys(display).map(val=>(
                      <span id={val} className='dpwords'
                      style={{
                        color:display[val].color.length>''?'white':'black',
                        backgroundColor:display[val].color.length==1?display[val].color[0]:'' ,
                        backgroundImage:display[val].color.length>1?'linear-gradient(45deg,'+display[val].color +')':'',
                    
                }}
                    
                    
                      onClick={()=>{
                          var jj=display
                        //   jj[val].color=color[tag]
                        jj[val].color.push(color[tag])
                          setdisplay(jj)
                          let vc=['red','green']
                          $('#'+val).css({
                            backgroundColor:display[val].color.length==1?display[val].color[0]:'' ,
                            backgroundImage:display[val].color.length>1?'linear-gradient(45deg,'+display[val].color +')':'',
                            color:'white'
                        })
                          num++;
                          if(num==3){
                              num=1;
                              k2=val;
                              var jj=data;
                              jj[k1].w2=jj[k2].w1;
                            //   jj[k2].w2=jj[k1].w1;
                              jj[k1].k1=k1;
                              jj[k1].k2=k2;
                            //   jj[k2].k1=k2;
                            //   jj[k2].k2=k1;
                              jj[k1].color=color[tag];
                            //   jj[k2].color=color[tag] ;
                              jj[k1].tags=tag
                            //   jj[k2].tags=tag
                            //   jj[k2].colors.push(color[tag])


                              color=null
                              setdata(jj)
                              var jl=display;
                            //   jl[k1].color.push(color[tag])
                            //   jl[k2].color.push(color[tag])

                              var tagval=                        {'prayojanam': 1, 'sampraxanam': 0, 'purvakalah': 0, 'samucciwam': 0, 'anuyogi': 0, 'karwqsamanaxikaranam': 0, 'bavalakranasapwami_samanakalah': 0, 'hewuh': 0, 'apaxanam': 0, 'root': 0, 'samboxyah': 0, 'upapaxasambanxah': 0, 'karma': 0, 'karmasamanaxikaranam': 0, 'axikaranam': 0, 'sambanxah': 0, 'waxarwyam': 0, 'kriyaviseranam': 0, 'prawiyogi': 0, 'karanam': 0, 'karwa': 0, 'viseranam': 0, 'rartisambanxah': 0}    
settags(tagval)
                          }
                          if(num==2){
                              k1=val;
                              if(data[k1].w2){
                                  var jj=display
                                  jj[data[k1].k2].color=[]
                                  setdisplay(jj)
                              }   
                          }

                      }}
                      >
                          {display[val].w1}
                          </span>    
                  ))
              }
              </div>
              {
                  Object.keys(data).map(val=>(
                      data[val].w2&&<Xarrow start={(data[val].k1).toString()} path={'grid'} end={(data[val].k2).toString()} curveness={2} color={color[data[val].tags]}
                      
                     startAnchor={'top'}
                     endAnchor={'top'  }
                     _cpy1Offset={-33-(Math.abs(data[val].k1-data[val].k2)*16+Math.max(data[val].k1,data[val].k2)*2)  }
                     _cpy2Offset={-33-(Math.abs(data[val].k1-data[val].k2)*16+Math.max(data[val].k1,data[val].k2)*2)}
                     headShape='circle'
                     headSize={2}
                     headColor="black"
                     arrowBodyProps={{
                         className:'' ,
                         style:{position:'relative',
                        },
                         onClick:()=>{
                         var jj=data,j=display,jl=[],l;
                         console.log(j,display)
                         for(l=0;l<display[data[val].k2].color.length;l++){
                             if(display[data[val].k2].color[l]==data[val].color)
                             break;
                             jl.push(display[data[val].k2].color[l])
                         }
                         for(l++;l<display[data[val].k2].color.length;l++){
                             jl.push(display[data[val].k2].color[l])
                         }
                         j[data[val].k2].color=jl
                         j[val].color=[]
                         jj[data[val].k1]={w1:data[val].w1,color:[]}
                         setdisplay(j)
                         setdata(jj)
                         var tagval=                        {'prayojanam': 1, 'sampraxanam': 0, 'purvakalah': 0, 'samucciwam': 0, 'anuyogi': 0, 'karwqsamanaxikaranam': 0, 'bavalakranasapwami_samanakalah': 0, 'hewuh': 0, 'apaxanam': 0, 'root': 0, 'samboxyah': 0, 'upapaxasambanxah': 0, 'karma': 0, 'karmasamanaxikaranam': 0, 'axikaranam': 0, 'sambanxah': 0, 'waxarwyam': 0, 'kriyaviseranam': 0, 'prawiyogi': 0, 'karanam': 0, 'karwa': 0, 'viseranam': 0, 'rartisambanxah': 0}    
                         settags(tagval)

                        },
                         cursor:'pointer',
                                
                     }}
                     arrowHeadProps={{
                         startOffset:data[val].k2*22
                     }}
                     labels={<div style={{color:'white',backgroundColor:color[data[val].tags],fontSize:11,borderRadius:6,padding:2,zIndex:11,
                    position:'absolute',bottom:(33+(Math.abs(data[val].k1-data[val].k2)*16+Math.max(data[val].k1,data[val].k2)*2))/7 ,

                    
                    }} >{data[val].tags} </div>}
                      /> 
                  ))
              }
          </div>
          {/* <div className='dpdisplay' >
              {
                  Object.keys(data).map(val=>(
                      <div style={{color:'black',display:'flex'}} >
                          <span style={{color:'black',padding:20}}  >{data[val].w1}</span>
                          <span style={{color:'black',padding:20}}  >{data[val].w2}</span>
                          <span style={{color:'black',alignSelf:'center',width:20,height:20,borderRadius:20 ,backgroundColor:data[val].color}}  >  </span>
                          </div>
                  ))
              }
          </div> */}
          <div style={{color:'black',
        position:'relative',top:'11%',padding:11,display:'flex',justifyContent:'center'
        }} >
            <button style={{color:'black',width:44,height:44,borderRadius:22,borderStyle:'solid',display:'flex',justifyContent:'center',backgroundColor:'blue'}} 
            onClick={()=>{
                if(dataval>1){
                    setdataval(dataval-1)
                    
                setdata(filedata[dataval])
                setdisplay(filedata[dataval])
                var jj=filedata[dataval]
                Object.keys(jj).map(val=>{
                    jj[filedata[dataval][val].k1].color.push(color[filedata[dataval][val].tags])
                    jj[filedata[dataval][val].k2].color.push(color[filedata[dataval][val].tags])
                })
                setdisplay(jj)
          
            }
            }}
            > <FaArrowLeft color='white' style={{color:'white',alignSelf:'center'}} /> </button>
            <span style={{color:'black',padding:11}} >{dataval} </span>
            <button style={{color:'black',width:44,height:44,borderRadius:22,borderStyle:'solid',display:'flex',justifyContent:'center',backgroundColor:'blue'}}
            onClick={()=>{
                setdataval(dataval+1)
                setdata(filedata[dataval])
                var jj=filedata[dataval]
                Object.keys(jj).map(val=>{
                    jj[filedata[dataval][val].k1].color.push(color[filedata[dataval][val].tags])
                    jj[filedata[dataval][val].k2].color.push(color[filedata[dataval][val].tags])
                })
                setdisplay(jj)
            }}
            ><FaArrowRight color='white' style={{color:'white',alignSelf:'center'}}
            
            />  </button>
          </div>
      </div>
  )
}

export default React.memo(OldDp)