import React from 'react'
import './Navbar.scss'
import {Link} from 'react-router-dom'
import { FaUser,FaBrain } from "react-icons/fa";

export const Navbar = () => {

  return (
    <div className="navbar">
        <div className="navbar-logo">
          <FaBrain />
            <Link to="/"><span style={{fontSize:"0.8em"}}>farm.</span><span style={{color:"#45C4B0"}}>AI</span><span style={{fontSize:"0.8em"}}>d</span></Link>
            {/* <Link to="/">mental.AId</Link> */}
        </div>
        <ul className='navbar-icons'>
          <li><Link to="/login"><FaUser /></Link></li>
        </ul>
    </div>
  )
}
