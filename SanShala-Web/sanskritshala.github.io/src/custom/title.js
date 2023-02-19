import React from 'react';
import {FaBlog, FaCopy, FaDochub, FaFacebook, FaFile, FaGoogle, FaHamburger, FaHome, FaInstagram, FaPagelines, FaReadme, FaRegCopyright, FaSnapchat, FaTeamspeak} from 'react-icons/fa'
import '../css/title.css'
import {BrowserRouter as Router, Routes, Route, Link, Outlet} from 'react-router-dom'
import Publications from '../pages/publications';
import Home from '../pages/home';
import Documentation from '../pages/documentation';
import Team from '../pages/team';
import $ from 'jquery'
function Title({width,page}) {
  return <div>
      <div className='title'>
          <div className='websitetitle' >SanskritShala</div>
          <div style={{display:'flex'}} >

          {<div display='false' className='pages' >
              <Link to='/' className='span' style={{color:page=='Home'?'blue':''}} ><FaHome style={{verticalAlign:'baseline'}} /> Home</Link>
              {/* <div onClick={()=>{window.open()}}  className='span' style={{color:page=='Documentation'?'blue':''}}><FaFile/>  Documentation</div> */}
              {/* <div  className='span' style={{color:page=='Blogs'?'blue':''}}onClick={()=>{window.open('https://agrawalanshul053.github.io/') }} ><FaBlog/> Blogs</div> */}
              <Link to='/resources' className='span' style={{color:page=='Resources'?'blue':''}}><FaReadme/> Resources</Link>
              <Link to='/publications' className='span' style={{color:page=='Publications'?'blue':''}}><FaPagelines/>  Publications</Link>
              <Link to='/team' className='span' style={{color:page=='Team'?'blue':''}}><FaTeamspeak/> Team</Link>
          </div> }
              <FaHamburger color='pink' className='titleicon' onClick={()=>{
                  $('.pages').attr(
                      'display',($('.pages').attr('display'))=='false'?'true':'false'
                  )
              }} />
          </div>
      </div>
      <Outlet/>
      
  </div>;
}

export default Title;
