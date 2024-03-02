import React from 'react'
import './Chat.scss'

export const Chat = ({chat}) => {
    console.log(chat)
  return (
    <div className={chat.author==="me"? "chat-bubble-container-right":"chat-bubble-container-left"}>
        <span>{chat.message}</span>
    </div>
  )
}
