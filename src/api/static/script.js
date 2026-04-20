document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const toggleBtns = document.querySelectorAll('.toggle-btn');
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const chatHistory = document.getElementById('chat-history');
    const sendBtn = document.getElementById('send-btn');
    
    // State
    let currentProvider = 'ollama';

    // --- Provider Toggle ---
    toggleBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            toggleBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentProvider = btn.dataset.provider;
        });
    });

    // --- Drag and Drop File Upload ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => handleFiles(e.dataTransfer.files));
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', function() { handleFiles(this.files); });

    async function handleFiles(files) {
        if (!files.length) return;
        const file = files[0];
        if (file.type !== 'application/pdf') {
            uploadStatus.innerHTML = `<span class="text-error">Error: Only PDFs are supported.</span>`;
            return;
        }

        uploadStatus.innerHTML = `<span class="text-dim">Uploading & Processing...</span>`;
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if (res.ok) {
                uploadStatus.innerHTML = `<span style="color: var(--success)">✅ Indexed ${data.pages_processed} pages (${data.chunks_added} chunks)</span>`;
            } else {
                uploadStatus.innerHTML = `<span style="color: var(--error)">❌ ${data.detail || 'Upload failed'}</span>`;
            }
        } catch (err) {
            uploadStatus.innerHTML = `<span style="color: var(--error)">❌ Connection error</span>`;
            console.error(err);
        }
    }

    // --- Chat Setup ---
    function appendMessage(text, isUser = false) {
        // Remove empty state if present
        const emptyState = chatHistory.querySelector('.empty-state');
        if (emptyState) emptyState.remove();

        const msgDiv = document.createElement('div');
        msgDiv.className = `msg ${isUser ? 'msg-user' : 'msg-ai'}`;
        
        // Simple line break parsing for AI responses
        if (isUser) {
            msgDiv.textContent = text;
        } else {
            // Very basic markdown to HTML for paragraphs
            const paragraphs = text.split('\n\n');
            msgDiv.innerHTML = paragraphs.map(p => {
                // simple bold parsing
                let parsed = p.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                return `<p>${parsed}</p>`;
            }).join('');
        }
        
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return msgDiv;
    }

    function appendTypingIndicator() {
        const emptyState = chatHistory.querySelector('.empty-state');
        if (emptyState) emptyState.remove();

        const indicator = document.createElement('div');
        indicator.className = 'msg msg-ai typing';
        indicator.innerHTML = '<span></span><span></span><span></span>';
        chatHistory.appendChild(indicator);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return indicator;
    }

    function appendCitations(msgDiv, citationsArray) {
        if (!citationsArray || citationsArray.length === 0) return;
        
        const citationContainer = document.createElement('div');
        citationContainer.className = 'citations';
        
        citationsArray.forEach((cit, index) => {
            const chip = document.createElement('span');
            chip.className = 'citation-chip';
            chip.title = `Score: ${cit.score}`;
            chip.textContent = `[${index + 1}] ${cit.pdf} (p.${cit.page})`;
            citationContainer.appendChild(chip);
        });

        msgDiv.appendChild(citationContainer);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    // --- Submit Query ---
    queryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = queryInput.value.trim();
        if (!text) return;

        // Reset input and lock btn
        queryInput.value = '';
        sendBtn.disabled = true;

        // Render user message
        appendMessage(text, true);

        // Render typing indicator
        const indicator = appendTypingIndicator();

        try {
            const res = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: text,
                    provider: currentProvider
                })
            });
            const data = await res.json();
            
            indicator.remove(); // Remove typing
            
            if (res.ok) {
                const newMsg = appendMessage(data.answer, false);
                appendCitations(newMsg, data.citations);
            } else {
                appendMessage(`Error: ${data.detail || 'Request failed'}`, false);
            }
        } catch (err) {
            indicator.remove();
            appendMessage(`Connection error`, false);
            console.error(err);
        }

        sendBtn.disabled = false;
        queryInput.focus();
    });
});
