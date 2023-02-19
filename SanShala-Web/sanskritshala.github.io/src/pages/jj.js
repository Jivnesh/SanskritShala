import React from 'react';
import { FaSearch } from 'react-icons/fa';
import Details from '../custom/details';
import Title from '../custom/title';

function JJ({title}) {
  return <div  >
      <Title/>
      <div style={{backgroundColor:'#1C2833',color:'white' ,height:400,width:'100%'}} >
          <div style={{color:'white',padding:10,fontSize:30}} >Word Segmentation</div>
          <p>
              Description about the functionality are displayed here
          </p>
          <p>Input Scheme <select>
              <option value={'input'}>Input</option>
              <option value={'value'}>Value</option>
              </select> Output Scheme <select>
              <option value={'output'}>Output</option>
              <option value={'value'}>Value</option>
                  
                  </select>  </p>
          <div style={{display:'flex', width:'45%',backgroundColor:'white', margin:'auto',borderRadius:60,marginTop:40}} >
              <FaSearch color='black' style={{alignSelf:'center',paddingTop:12,paddingBottom:12,marginLeft:30}} />
              <input style={{border:'none',flex:1,display:'flex',marginLeft:10,borderRadius:60,backgroundColor:'transparent',
                outline:'none',':hover':{color:'white'}
            }} />
          </div>
          <button style={{
            padding:10,margin:10,borderRadius:10,outline:'none',
            fontWeight:'bold',color:'white',marginTop:40,backgroundColor:'transparent',
            borderStyle:'solid',borderColor:'white'
        }} >Submit</button>

      </div>
  </div>;
}

export default JJ;
