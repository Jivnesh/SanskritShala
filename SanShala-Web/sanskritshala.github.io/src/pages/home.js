import React,{useState} from 'react';
import '../css/home.css'
import Cards from '../custom/cards'
import NewCard from '../custom/Newcard'
import {FaBlog, FaCopy, FaDochub, FaFacebook, FaFile, FaGoogle, FaHamburger, FaHome, FaInstagram, FaPagelines, FaReadme, FaRegCopyright, FaSnapchat, FaTeamspeak} from 'react-icons/fa'
import Details from '../custom/details'
import {slide as Menu} from 'react-burger-menu'
import {  useNavigate } from 'react-router-dom';
import SideBar from "../custom/sidebar";
import Title from '../custom/title.js'
import Chatbox from '../custom/chatbox';
function Home({width}) {
  const [cbox,setcbox]=useState(1);
    const [val,setval]=useState({})
    var [cc,setcc]=useState([{
        bot:'Hi! I\'m a Bot. Let me know if you have any questions regarding our tool!',
    },
    {bot:'Can you please let me know how can I help you?'
    },
    ]);
    let navigate=useNavigate();
    function Nav(){
        navigate('/')
    }
  return <div className='home' >
      <Title page={'Home'} />
      <br/>
      <div className='content'>
              <div style={{display:'inline-block'}} >
                  <NewCard onClick={Nav} vid='tile7' title="About SanskritShala" vcolor1="rgba(241, 196, 15)" vcolor2="rgba(252, 243, 207, 0.041)" />
              </div>
        </div>
      <div className='content'>
              <div style={{display:'inline-block'}} >
                  <Cards onClick={()=>{window.location.href='http://cnerg.iitkgp.ac.in/translist/'}}  vid='tile1' title="Word Segmentation" vcolor1="rgba(243, 156, 18 )" vcolor2="rgba(217, 235, 140, 0.041)" />
                  <Cards onClick={()=>{window.location.href='http://cnerg.iitkgp.ac.in/tramp/'}} vid='tile2' title="Morphological Tagging" vcolor1="rgba(187, 143, 206)" vcolor2="rgba(232, 211, 143, 0.041)" />
                  <Cards onClick={()=>{window.location.href='http://cnerg.iitkgp.ac.in/sandp/'}} vid='tile3' title="Dependency Parsing" vcolor1="rgba(69, 179, 157)" vcolor2="rgba(209, 242, 235, 0.041)" />
                  {/* <Cards onClick={Nav} vid='tile4' title="Anvaya System" vcolor1="rgba(241, 196, 15)" vcolor2="rgba(252, 243, 207, 0.041)" /> */}
                  {/* <Cards onClick={()=>{window.location.href='https://sanskritshala.github.io/shloka.github.io/'}} vid='tile5' title="Shloka" vcolor1="rgba(93, 173, 226)" vcolor2="rgba(214, 234, 248, 0.041)" /> */}
                  <Cards onClick={()=>{window.location.href='http://cnerg.iitkgp.ac.in/sacti/'}} vid='tile6' title="Compound Classifier" vcolor1="rgba(203, 67, 53)" vcolor2="rgba(245, 183, 177, 0.041)" />
                  {/* <Cards onClick={Nav} vid='tile7' title="Documentation" vcolor1="rgba(121,132,112)" /> */}

              </div>
        </div>
        <Chatbox setval={setcbox} val={cbox} cc={cc} setcc={setcc}/>
        <br/>
        <Details/>
        
  </div>;
}

export default Home;