// Загрузка библиотеки marked.js
const script = document.createElement('script');
script.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
script.onload = () => {
    console.log('marked.js загружен!');
    // После загрузки библиотеки можно использовать её функции
    // Настройка marked.js
    marked.setOptions({
        breaks: true, // Перенос строки как <br>
//        gfm: true, // Поддержка GitHub Flavored Markdown
    })};
document.head.appendChild(script);

async function load_History_toChat() {
    try {
        const response = await fetch('http://127.0.0.1:5000/history?user_id=default_user');
        if (!response.ok) {
            throw new Error('Ошибка при загрузке истории');
        }
        const data = await response.json();
        const chatBox = document.getElementById('chat-box');

        // Очищаем чат перед загрузкой истории
        chatBox.innerHTML = '';

        // Добавляем каждое сообщение в чат
        data.history.forEach(message => {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${message.role === 'Пользователь' ? 'user-message' : 'bot-message'}`;

        // Преобразуем Markdown в HTML
        const messageContent = marked.parse(message.message);

        messageElement.innerHTML = `
            <img src="http://127.0.0.1:5000/images/${message.role === 'Пользователь' ? 'human.png' : 'llama.png'}" alt="${message.role === 'Пользователь' ? 'User' : 'Bot'}" class="${message.role === 'Пользователь' ? 'user-icon' : 'bot-icon'}">
            ${messageContent}
        `;
        chatBox.appendChild(messageElement);
    });

        // Прокручиваем чат вниз
        chatBox.scrollTop = chatBox.scrollHeight;
    } 
    catch (error) {
        console.error('Ошибка при загрузке истории:', error);
    }
}

// Отправка сообщения
async function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (!userInput) return;

    // Добавляем сообщение пользователя в чат
    const chatBox = document.getElementById('chat-box');
    const userMessage = document.createElement('div');
    userMessage.className = 'message user-message';

    // Преобразуем Markdown в HTML
    const userMessageContent = marked.parse(userInput);

    userMessage.innerHTML = `
        <img src="http://127.0.0.1:5000/images/human.png" alt="User" class="user-icon">
        ${userMessageContent}
    `;
    chatBox.appendChild(userMessage);

    // Очищаем поле ввода
    document.getElementById('user-input').value = '';

    // Отправляем сообщение на сервер
    try {
        const response = await fetch('http://127.0.0.1:5000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: userInput }),
        });

        if (!response.ok) {
            throw new Error('Ошибка при отправке сообщения');
        }

        const data = await response.json();

        // Добавляем ответ бота в чат
        const botMessage = document.createElement('div');
        botMessage.className = 'message bot-message';

        // Преобразуем Markdown в HTML
        const botMessageContent = marked.parse(data.response);

        botMessage.innerHTML = `
            <img src="http://127.0.0.1:5000/images/llama.png" alt="Bot" class="bot-icon">
            ${botMessageContent}
        `;

        chatBox.appendChild(botMessage);

        // Прокручиваем чат вниз
        chatBox.scrollTop = chatBox.scrollHeight;
    } 
    catch (error) {
        console.error('Ошибка:', error);
        alert('Произошла ошибка при отправке сообщения');
    }
}

// Отправка сообщения по нажатию Enter
document.getElementById('user-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Загружаем историю при загрузке страницы
window.onload = load_History_toChat;

// Отправка запроса на завершение работы сервера при закрытии окна
window.addEventListener('beforeunload', async () => {
    try {
        await fetch('http://127.0.0.1:5000/shutdown', {
            method: 'POST',
        });
    } catch (error) {
        console.error('Ошибка при отправке запроса на завершение:', error);
    }
});