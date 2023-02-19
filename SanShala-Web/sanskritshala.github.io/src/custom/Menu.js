import React, { useState } from 'react'
import {FaArrowUp,FaArrowDown} from 'react-icons/fa'
function Menu({options}) {
    const [toggle,settoggle]=useState(0)
  return (
      <div style={{color:'black'}} >
          <div className='menulist' style={{color:'white',display:'flex',justifyContent:'center' ,height:'100%',backgroundColor:'blue',cursor:'pointer'}} onClick={()=>{
              settoggle((!toggle))
          }} >
              {toggle!=0?<FaArrowUp style={{color:'white',alignSelf:'center'}} />:
              <FaArrowDown style={{color:'white',alignSelf:'center'}}/> 
              }              
          </div>
      </div>
  )
}

export default Menu