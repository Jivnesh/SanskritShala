import React from 'react';
import Title from '../custom/title'
import '../css/team.css'
import sir from '../images/sir.jpeg'
import pawangoyal from '../images/pawangoyal.jpg'
import TeamTile from '../custom/teamtile.js'
function team({width}) {
  return <div>
    <Title width={width} page={"Team"} />
    <div style={{padding:10}} >
      
    <div className='team' >
      Professors
    </div>
    <div style={{display:'flex',justifyContent:'center'}} >
    <TeamTile image={sir} member={{name:'Dr. Laxmidhar Behera',description:'Professor',email:'IIT Kanpur',mobile:'mobile',website:'https://home.iitk.ac.in/~lbehera/'}} />
    <TeamTile image={pawangoyal} member={{name:'Dr. Pawan Goyal',description:'Associate Professor',email:'IIT Kharagpur',mobile:'mobile',website:'https://cse.iitkgp.ac.in/~pawang/'}} />
    </div>
    <div className='team'>
      Contributors
      
    </div>
    <div style={{color:'grey',textAlign:'left',fontSize:20,marginLeft:10}} >
      Phd Students
    </div>
    <div style={{textAlign:'left',paddingTop:20}} >
    <div className='listnames' >1. Jivnesh Sandhan, IIT Kanpur </div>
    <div className='listnames' >2. Rathin Singha, UCLA </div>
    <div className='listnames' >3. Amrith Krishna, Uniphore </div>
    </div>
    <div style={{color:'grey',textAlign:'left',fontSize:20,marginLeft:10,marginTop:30}} >
      M.Tech Students
    </div>
    <div style={{textAlign:'left',paddingTop:20}} >
    <div className='listnames' >1. Narein Rao, IIT Kanpur </div>
    </div>
    <div style={{color:'grey',textAlign:'left',fontSize:20,marginLeft:10,marginTop:30}} >
      B.Tech Students
    </div>
    <div style={{textAlign:'left',paddingTop:20}} >
    <div className='listnames' >1. Anshul Agarwal, IIT Kanpur </div>
    <div className='listnames' >2. Hrithik Sharma, IIT Kanpur </div>
    <div className='listnames' >3. Ashish Gupta, IIT Kharagpur </div>
    <div className='listnames' >4. Ayush Daksh, IIT Kharagpur </div>
    </div>
    </div>

  </div>;
}

export default team;
