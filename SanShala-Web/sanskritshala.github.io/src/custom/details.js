import React from 'react';
import {FaCopy, FaFacebook, FaGoogle, FaInstagram, FaRegCopyright, FaSnapchat} from 'react-icons/fa'
import '../css/details.css'
function Details() {
  return <div className='details' >
  <div className='information' >
      <div className='detailcards' >
          <div style={{width:'100%'}} >
              
          <div>
              Web Team
          </div>
          <div className='detaildescription' >
              <div>Anshul Agarwal</div>
              <div>Hritik Sharma</div>
          </div>
          </div>
      </div>
      <div className='detailcards' >
          
      <div style={{width:'100%'}}>
              
              <div>
                  Contact
              </div>
              <div className='detaildescription' >
              <div>Jivnesh Sandhan </div>
              <div>Email: jivneshsandhan@gmail.com</div>
              
                  
              </div>
              </div>
          </div>
      <div className='detailcards' >
      <div style={{width:'100%'}} >
              
              <div>
                  Follow Us
              </div>
              <div className='detaildescription' >
                  <div style={{display:'flex',justifyContent:'space-around'}} >
                  <FaFacebook className='icons' />
                  <FaGoogle className='icons'/>
                  <FaInstagram className='icons'/>
                  <FaSnapchat className='icons'/>
                  </div>
              </div>
              </div>
          </div>
  </div>
  <div style={{display:'flex',flexDirection:'row-reverse'}} >
      <div style={{alignSelf:'center',padding:10}} >
          This template is made by Hrithik Sharma
      </div>
      <FaRegCopyright style={{alignSelf:'center'}} />
  </div>
</div>;
}

export default Details;
