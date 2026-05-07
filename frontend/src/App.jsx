import React, { useEffect, useMemo, useRef, useState } from "react";

const PROFILE = {
  name: "Nour Shraif",
  title: "Computer Science Graduate",
  tagline: "I build practical ML and software projects with a focus on useful outcomes.",
  links: [
    { label: "LinkedIn", href: "https://www.linkedin.com/" },
    { label: "GitHub", href: "https://github.com/" },
    { label: "Email", href: "mailto:nourshraif4@gmail.com" },
  ],
};

const INITIAL_BOT_MESSAGE = {
  id: "welcome",
  role: "bot",
  text: "Hi, I am Nour's resume assistant. Ask me anything about experience, projects, or skills.",
  createdAt: new Date().toISOString(),
};

function formatHoverTimestamp(iso) {
  const date = new Date(iso);
  try {
    return date.toLocaleString([], {
      dateStyle: "medium",
      timeStyle: "short",
    });
  } catch {
    // Fallback for browsers that do not support dateStyle/timeStyle options.
    return `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
  }
}

async function sendChatMessage(message) {
  const baseUrl = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
  const endpoint = baseUrl ? `${baseUrl}/api/chat` : "/api/chat";
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    const maybeJson = await response.json().catch(() => null);
    const detail = maybeJson?.error || maybeJson?.message || `HTTP ${response.status}`;
    throw new Error(detail);
  }

  const data = await response.json();
  return data.reply || data.answer || "I did not get a response body from the server.";
}

export default function App() {
  const [messages, setMessages] = useState([INITIAL_BOT_MESSAGE]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const listRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    const container = listRef.current;
    if (!container) return;
    container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const canSubmit = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  const makeId = () =>
    (globalThis.crypto && "randomUUID" in globalThis.crypto)
      ? globalThis.crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;

  const handleSubmit = async (event) => {
    event.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    const userMessage = {
      id: makeId(),
      role: "user",
      text: question,
      createdAt: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const reply = await sendChatMessage(question);
      setMessages((prev) => [
        ...prev,
        {
          id: makeId(),
          role: "bot",
          text: reply,
          createdAt: new Date().toISOString(),
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: makeId(),
          role: "bot",
          text: `I hit an issue reaching the backend: ${error.message}`,
          createdAt: new Date().toISOString(),
        },
      ]);
    } finally {
      setLoading(false);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  };

  return (
    <div className="app-shell">
      <div className="texture" aria-hidden="true" />

      <section className="profile-panel" aria-label="Profile summary">
        <div className="profile-content">
          <h1>{PROFILE.name}</h1>
          <p className="title">{PROFILE.title}</p>
          <p className="tagline">{PROFILE.tagline}</p>

          <nav className="quick-links" aria-label="Quick links">
            {PROFILE.links.map((link) => (
              <a key={link.label} href={link.href} target="_blank" rel="noreferrer">
                {link.label}
              </a>
            ))}
          </nav>
        </div>
      </section>

      <section className="chat-panel" aria-label="Chat interface">
        <div className="chat-header">
          <h2>Resume Conversation</h2>
          <p>Ask me anything about my experience...</p>
        </div>

        <div className="chat-messages" ref={listRef}>
          {messages.map((message) => (
            <article
              key={message.id}
              className={`message-row ${message.role === "user" ? "user" : "bot"}`}
            >
              <div className="message-bubble" title={formatHoverTimestamp(message.createdAt)}>
                {message.text}
              </div>
            </article>
          ))}

          {loading && (
            <article className="message-row bot">
              <div className="message-bubble typing" title="Typing...">
                <span />
                <span />
                <span />
              </div>
            </article>
          )}
        </div>

        <form className="chat-input-wrap" onSubmit={handleSubmit}>
          <input
            ref={inputRef}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask me anything about my experience..."
            disabled={loading}
            autoFocus
          />
          <button type="submit" disabled={!canSubmit} aria-label="Send message">
            <svg viewBox="0 0 24 24" role="img" aria-hidden="true">
              <path d="M4 12h14" />
              <path d="m12 5 7 7-7 7" />
            </svg>
          </button>
        </form>
      </section>
    </div>
  );
}
