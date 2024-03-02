import { useState } from 'react'
import { Routes, Route } from "react-router-dom"
import { Home } from './pages/Home'
import { Navbar } from './pages/Navbar'
import './App.css'

function App() {
  

  return (
    <div className="app">
      <Navbar />
      <Routes>
          <Route path="/" element={<Home />} />
      </Routes>
    </div>
  )
}

export default App
