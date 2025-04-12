import React, { useState, useRef, useEffect } from 'react';
import Message from './Message';

const Chat = ({ messages, onSendMessage, isLoading }) => {
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim()) {
      onSendMessage(inputValue);
      setInputValue('');
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="welcome-message">
            <h3>Welcome to MoodTunes!</h3>
            <p>Tell me how you're feeling or what you're doing, and I'll recommend the perfect music for you.</p>
            <p>Try something like:</p>
            <ul>
              <li>"I'm feeling happy and energetic today"</li>
              <li>"I need relaxing music for studying"</li>
              <li>"Songs for a rainy day"</li>
            </ul>
          </div>
        ) : (
          messages.map((message, index) => (
            <Message 
              key={index} 
              text={message.text} 
              sender={message.sender} 
              isReason={message.isReason} 
            />
          ))
        )}
        {isLoading && (
          <div className="message bot">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form onSubmit={handleSubmit} className="chat-input">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Describe your mood or activity..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !inputValue.trim()}>
          <i className="fas fa-paper-plane"></i>
        </button>
      </form>
    </div>
  );
};

export default Chat;