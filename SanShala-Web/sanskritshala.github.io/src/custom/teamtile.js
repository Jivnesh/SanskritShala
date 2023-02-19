import React from 'react';
import '../css/teamtile.css'
import wallpaper from '../images/wallpaper.jpg'
function teamtile({image,member,...props}) {
  return <div className='teamtile'  >
      <img src={image} className='imageteamtile' />
      <div className='teamtiledetails' >
          <div>{member.name} </div>
          <div>{member.description} </div>
          <div>{member.email} </div>
          <a href={member.website}>Home </a>
      </div>
  </div>;
}
export default teamtile;