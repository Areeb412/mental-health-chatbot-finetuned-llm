// ───────────────────────────────────────────────────────────────
// DOM Elements
// ───────────────────────────────────────────────────────────────
console.log('🚀 Chat.js initializing...');

const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const typingIndicator = document.getElementById('typingIndicator');
const charCount = document.getElementById('charCount');
const infoBtn = document.getElementById('infoBtn');
const modal = document.getElementById('infoModal');
const closeBtn = document.querySelector('.close');

console.log('✅ Elements loaded:');
console.log('  - chatMessages:', !!chatMessages);
console.log('  - userInput:', !!userInput);
console.log('  - sendBtn:', !!sendBtn);
console.log('  - typingIndicator:', !!typingIndicator);
console.log('  - charCount:', !!charCount);
console.log('  - infoBtn:', !!infoBtn);
console.log('  - modal:', !!modal);
console.log('  - closeBtn:', !!closeBtn);

// ───────────────────────────────────────────────────────────────
// Event Listeners
// ───────────────────────────────────────────────────────────────

// Character counter (if element exists)
if (charCount) {
    userInput.addEventListener('input', () => {
        const count = userInput.value.length;
        charCount.textContent = count;
        
        // Auto-resize textarea
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
    });
} else {
    // Just handle auto-resize without character counter
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
    });
}

// Send button click handler
if (sendBtn) {
    sendBtn.addEventListener('click', (e) => {
        e.preventDefault();
        console.log('✅ Send button clicked');
        sendMessage();
    });
} else {
    console.error('❌ Send button not found!');
}

// Send on Enter, new line on Shift+Enter or Ctrl+Enter
userInput.addEventListener('keydown', (e) => {
    if ((e.key === 'Enter' && !e.shiftKey) || (e.ctrlKey && e.key === 'Enter')) {
        e.preventDefault();
        sendMessage();
    }
});

// Modal events (if modal exists)
if (infoBtn && modal && closeBtn) {
    infoBtn.addEventListener('click', () => {
        modal.classList.add('active');
    });

    closeBtn.addEventListener('click', () => {
        modal.classList.remove('active');
    });

    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('active');
        }
    });
}

// ───────────────────────────────────────────────────────────────
// Core Chat Functions
// ───────────────────────────────────────────────────────────────

async function sendMessage() {
    console.log('📤 sendMessage() called');
    const message = userInput.value.trim();
    console.log('Message content:', message);
    
    if (!message) {
        console.warn('⚠️ Empty message, not sending');
        return;
    }
    
    console.log('🔒 Disabling send button');
    if (sendBtn) sendBtn.disabled = true;

    // Display user message
    console.log('💬 Appending user message to chat');
    appendMessage(message, 'user');

    // Clear input
    userInput.value = '';
    userInput.style.height = 'auto';
    if (charCount) charCount.textContent = '0';

    // Show typing indicator
    console.log('⏳ Showing typing indicator');
    showTyping(true);

    try {
        console.log('📡 Sending fetch request to /get');
        const response = await fetch('/get', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ msg: message }),
        });

        console.log('📨 Response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        console.log('✅ Response received:', data);
        const isCrisis = data.is_crisis === true;
        appendMessage(data.response, 'bot', isCrisis);

    } catch (error) {
        console.error('❌ Error in sendMessage:', error);
        appendMessage(
            "I'm experiencing a connection issue. Please try again.",
            'bot'
        );
    } finally {
        console.log('🔓 Re-enabling send button and hiding typing');
        showTyping(false);
        if (sendBtn) sendBtn.disabled = false;
        userInput.focus();
    }
}

function appendMessage(text, sender, isCrisis = false) {
    console.log(`📌 appendMessage(${sender}):`, text.substring(0, 50) + '...');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${sender}-message`);

    // Create bubble
    const bubble = document.createElement('div');
    bubble.classList.add('bubble', `${sender}-bubble`);
    if (isCrisis) bubble.classList.add('crisis');

    // Add text
    const p = document.createElement('p');
    p.textContent = text;
    bubble.appendChild(p);

    // Add timestamp
    const timestamp = document.createElement('div');
    timestamp.classList.add('timestamp');
    timestamp.textContent = getTime();
    bubble.appendChild(timestamp);

    // Add avatar for bot messages
    if (sender === 'bot') {
        const avatar = document.createElement('div');
        avatar.classList.add('bot-avatar');
        avatar.innerHTML = '<i class="fas fa-heart"></i>';
        messageDiv.appendChild(avatar);
    }

    messageDiv.appendChild(bubble);
    
    // Append before typing indicator (now that it's inside chatMessages)
    if (typingIndicator && typingIndicator.parentNode === chatMessages) {
        console.log('✅ Inserting before typing indicator');
        chatMessages.insertBefore(messageDiv, typingIndicator);
    } else {
        console.log('✅ Appending message (typing indicator not found as child)');
        chatMessages.appendChild(messageDiv);
    }
    
    scrollToBottom();
}

function showTyping(visible) {
    console.log('⏳ showTyping:', visible);
    typingIndicator.style.display = visible ? 'flex' : 'none';
    if (visible) scrollToBottom();
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function getTime() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}