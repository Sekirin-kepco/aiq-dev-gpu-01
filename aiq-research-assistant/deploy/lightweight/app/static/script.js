// チャット UI のスクリプト

const messagesContainer = document.getElementById('chat-messages');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const loadingIndicator = document.getElementById('loading-indicator');
const statusIndicator = document.getElementById('status-indicator');

// ステータスをチェック
async function checkStatus() {
    try {
        const response = await fetch('/health');
        if (response.ok) {
            const data = await response.json();
            updateStatus(true, data.aira_available);
        } else {
            updateStatus(false);
        }
    } catch (error) {
        console.error('Status check failed:', error);
        updateStatus(false);
    }
}

// ステータス表示を更新
function updateStatus(connected, airaAvailable = false) {
    statusIndicator.textContent = connected 
        ? (airaAvailable ? '✓ AIRA 利用可能' : '✓ ローカルインデックス')
        : '✗ 接続不可';
    statusIndicator.className = 'status-indicator ' + (connected ? 'connected' : 'error');
}

// メッセージ送信
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    // ユーザーメッセージを表示
    addMessage(message, 'user');
    messageInput.value = '';
    messageInput.focus();

    // 送信ボタンを無効化
    sendButton.disabled = true;
    loadingIndicator.style.display = 'flex';

    try {
        // /query エンドポイントに POST
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ q: message })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // レスポンスに応じてメッセージを表示
        // Prefer the standardized `sources` field (server returns normalized hits there)
        if (data.sources && Array.isArray(data.sources) && data.sources.length > 0) {
            // Display generated answer first if present
            if (data.answer) {
                addMessage(data.answer, 'assistant', data.source, data.sources, data.key_points);
            } else {
                addMessage('関連ドキュメントが見つかりました。', 'assistant', data.source, data.sources, data.key_points);
            }
        } else if (data.hits && Array.isArray(data.hits) && data.hits.length > 0) {
            // backward compatibility: older responses may include `hits`
            if (data.answer) {
                addMessage(data.answer, 'assistant', data.source, data.hits, data.key_points);
            } else {
                addMessage('関連ドキュメントが見つかりました。', 'assistant', data.source, data.hits, data.key_points);
            }
        } else if (data.answer) {
            // 通常の回答
            addMessage(data.answer, 'assistant', data.source, null, data.key_points);
        } else {
            addMessage('応答の処理に失敗しました', 'assistant', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage(`エラー: ${error.message}`, 'assistant', 'error');
    } finally {
        sendButton.disabled = false;
        loadingIndicator.style.display = 'none';
    }
}

// メッセージを UI に追加（改良版：key_points と構造化レスポンスに対応）
function addMessage(text, sender = 'assistant', source = '', hits = null, keyPoints = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    const textDiv = document.createElement('div');
    const pElement = document.createElement('p');
    pElement.textContent = text;
    textDiv.appendChild(pElement);

    // 出典表示（assistant のメッセージのみ）
    if (source && sender !== 'user') {
        const sourceDiv = document.createElement('div');
        sourceDiv.className = 'message-source';
        sourceDiv.textContent = `📌 出典: ${source}`;
        textDiv.appendChild(sourceDiv);
    }

    // キーポイント表示（RAG応答用）
    if (keyPoints && Array.isArray(keyPoints) && keyPoints.length > 0 && sender !== 'user') {
        const keyPointsDiv = document.createElement('div');
        keyPointsDiv.className = 'message-key-points';
        keyPointsDiv.innerHTML = '<strong>📍 重要なポイント:</strong><ul>';
        keyPoints.forEach((point) => {
            const li = document.createElement('li');
            li.textContent = point;
            keyPointsDiv.querySelector('ul').appendChild(li);
        });
        keyPointsDiv.innerHTML += '</ul>';
        textDiv.appendChild(keyPointsDiv);
    }

    // 検索結果表示（hits/sources）
    if (hits && Array.isArray(hits) && hits.length > 0) {
        const hitsDiv = document.createElement('div');
        hitsDiv.className = 'message-hits';
        hitsDiv.innerHTML = '<strong>🔍 検索結果:</strong><br>';

        hits.forEach((hit, index) => {
            const hitItem = document.createElement('div');
            hitItem.className = 'message-hit';
            // hit is expected to be normalized: {id,file,page,snippet,text,score,source}
            const file = hit.file ? `<strong>${escapeHtml(String(hit.file))}</strong>` : '';
            const page = hit.page ? ` (ページ ${escapeHtml(String(hit.page))})` : '';
            const snippet = escapeHtml(hit.snippet || hit.text || '');
            const score = (hit.score !== undefined && hit.score !== null) ? ` <span class="message-hit-score">(スコア: ${Number(hit.score).toFixed(2)})</span>` : '';

            // Build controls: open link (if file) and toggle full snippet
            const openLink = hit.file ? `<a class="hit-open" href="/docs/${encodeURIComponent(hit.file)}" target="_blank" rel="noopener">📄 ソースを開く</a>` : '';
            const toggleBtn = `<button class="hit-toggle" data-hit="${index}">全文表示</button>`;
            hitItem.innerHTML = `
                <div style="display:flex;gap:8px;align-items:center;justify-content:space-between;">
                    <div><strong>結果 ${index + 1}:</strong> ${file}${page}${score}</div>
                    <div style="display:flex;gap:8px;align-items:center;">${openLink}${toggleBtn}</div>
                </div>
                <div class="hit-snippet" id="hit-snippet-${index}">${snippet}</div>
            `;
            // After inserting, attach click handler to toggle full snippet
            setTimeout(() => {
                const btn = hitItem.querySelector('.hit-toggle');
                if (btn) {
                    const snippetEl = hitItem.querySelector(`#hit-snippet-${index}`);
                    const fullText = snippetEl.textContent || snippetEl.innerText || '';
                    const shortText = fullText.length > 300 ? fullText.slice(0, 300) + '…' : fullText;
                    // initialize to short
                    snippetEl.textContent = shortText;
                    btn.addEventListener('click', () => {
                        if (btn.dataset.expanded === '1') {
                            snippetEl.textContent = shortText;
                            btn.textContent = '全文表示';
                            btn.dataset.expanded = '0';
                        } else {
                            snippetEl.textContent = fullText;
                            btn.textContent = '折りたたむ';
                            btn.dataset.expanded = '1';
                        }
                    });
                }
            }, 0);
            hitsDiv.appendChild(hitItem);
        });

        textDiv.appendChild(hitsDiv);
    }

    messageDiv.appendChild(textDiv);
    messagesContainer.appendChild(messageDiv);

    // 最下部にスクロール
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// HTML エスケープ関数
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// イベントリスナー
sendButton.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// 初期化
window.addEventListener('load', () => {
    checkStatus();
    messageInput.focus();
});

// 定期的にステータスをチェック
setInterval(checkStatus, 30000);
