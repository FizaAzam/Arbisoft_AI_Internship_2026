// http://localhost:5173/
import { useState, useRef, useEffect } from "react";
const API_URL = "http://127.0.0.1:8000";

export default function App() {
  //the conversation shown on screen: { role: "customer" | "assistant", text, routedTo }
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  //the backend gives us a session id on the first reply; we send it back every time
  const [sessionId, setSessionId] = useState(null);

  const [isWaiting, setIsWaiting] = useState(false);
  const [error, setError] = useState(null);

  //used to scroll the newest message into view
  const bottomRef = useRef(null);

  useEffect(() => { //for scrolling down whenever message/iswaitig changes
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isWaiting]);

  async function sendMessage() {
    const text = input.trim(); //removes spaces from the beginning/end
    if (!text || isWaiting) return; //Stop the function if: text is empty, or the chatbot is already waiting for a response.

    setMessages((old) => [...old, { role: "customer", text }]);  //Adds the customer's new message to the existing messages.
    setInput("");//clear the typing box for user
    setError(null); //remove any error mssgs
    setIsWaiting(true);  //sets the app as waiting for backend response

    try {//fetch() is the browser function used to make HTTP requests.
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST", //sending a chat mssg to backend
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });

      if (!response.ok) {//if http returned an error
        const problem = await response.json().catch(() => ({}));
        throw new Error(problem.detail || `Server returned ${response.status}`);
      }

      const data = await response.json(); //converting backend response to a json object
      setSessionId(data.session_id);
      setMessages((old) => [//adds the assistant's reply
        ...old,
        { role: "assistant", text: data.reply, routedTo: data.routed_to },
      ]);
    } catch (problem) {
      setError(problem.message);
    } finally {
      setIsWaiting(false);
    }
  }

  //handleKeyDown() is the function that decides what to do when a keyboard key is pressed in the textarea.
  //event contains info about the key press
  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      //if user pressed enter and didnt press shift then send the message
      event.preventDefault();//prevents Enter's normal behavior which is to insert a new line(ik default should  be sending but its moving to new line)
      sendMessage();
    }
  }



//This whole return (...) is the UI description of your React component.
//React reads it and renders that structure in the browser.
//basicaly describing how browser should look like
  return (
    <div className="app">
      <header className="header">
        <h1>Restaurant Assistant</h1>
        <p>Ask about the menu, or place an order.</p>
      </header>

      <main className="chat">
        {messages.length === 0 && !isWaiting && (
          <div className="empty">
            <p>Try one of these:</p>
            <ul>
              <li>"what do you have on the menu?"</li>
              <li>"add mango shake for 250"</li>
              <li>"I'm Ali and I'd like two chicken biryanis"</li>
            </ul>
          </div>
        )}

        {messages.map((message, index) => (
          <div key={index} className={`bubble ${message.role}`}>
            {message.routedTo && (
              <span className="tag">handled by: {message.routedTo}</span>
            )}
            <p>{message.text}</p>
          </div>
        ))}

        {isWaiting && (
          <div className="bubble assistant thinking">
            <p>thinking…</p>
          </div>
        )}

        {error && <div className="error">{error}</div>}

        <div ref={bottomRef} />
      </main>

      <footer className="composer">
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message…"
          rows={1}
          disabled={isWaiting}
        />
        <button onClick={sendMessage} disabled={isWaiting || !input.trim()}>
          Send
        </button>
      </footer>
    </div>
  );
}
