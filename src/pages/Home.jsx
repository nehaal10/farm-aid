import React, { useEffect, useState } from 'react'
import { FaTrash,FaPlay } from "react-icons/fa";
import "./Home.scss"
import { Chat } from '../components/Chat';
import axios from 'axios'

export const Home = () => {
    const [messageText, setMessageText] = useState("")
    const [chats, setChats] = useState([])
    const input = document.getElementById('chat-message')

    useEffect(()=>{
        console.log(chats)
        let msgSpace = document.getElementById('message-space')
        msgSpace.scrollTop = msgSpace.scrollHeight
    },[chats])

    const handleKeyPress = (e)=>{
        if(e.keyCode === 13){
            handleSend()
        }
    }

    const handleSend = async ()=>{
        setChats([...chats,{"author":"me","message":messageText}])

        input.value = ''
        // fetch("/user-query",{
        //     method:'GET',
        //     mode:'cors',
        //     headers:{'Content-Type':'application/json'}
        // })
        // .then((response)=>setChats([...chats,[response.data]]))
        axios.post("http://localhost:5000/user-query",{
            "question":messageText
        })
        // .then((response)=>console.log(response.data))
        .then((response)=>setChats([...chats,{"author":"me","message":messageText},{"author":"ai","message":response.data.answer}]))
        .catch((error)=>console.log(error))
    }
  return (
    <div className="home-container">
        <div className="chat-container">
            <div className="messages-container">
                <div className="message-space" id='message-space'>
                    {chats.map((chat, i)=>(
                        <Chat chat={chat} key={i} />
                    ))}
                </div>
            </div>
            <div className="user-control">
                <div className="delete-messages">
                    <FaTrash onClick={()=>{setChats([])}}/>
                </div>
                <div className="message-box">
                    <input type="text" name="chat-message" id="chat-message" onChange={(e)=>setMessageText(e.target.value)} onKeyDown={handleKeyPress}/>
                </div>
                <div className="send-message">
                    <FaPlay onClick={handleSend}/>
                </div>
            </div>
        </div>
    </div>
  )
}
