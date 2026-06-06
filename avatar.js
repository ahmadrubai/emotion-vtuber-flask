let currentEmotion = null;
let idleDimTimer = null;

const IDLE_DIM_DELAY_MS = 1200;
const NEUTRAL_AVATAR_SRC = "/static/assets/neutral.png";

function setAvatarIdleDim(isDimmed) {
    const avatarImage = document.getElementById("avatar-image");

    if (!avatarImage) {
        return;
    }

    avatarImage.classList.toggle("idle-dim", isDimmed);
}

function resetIdleDimTimer() {
    clearTimeout(idleDimTimer);
    setAvatarIdleDim(false);

    idleDimTimer = setTimeout(() => {
        const avatarImage = document.getElementById("avatar-image");

        if (avatarImage && avatarImage.classList.contains("meme-bounce")) {
            return;
        }

        setAvatarIdleDim(true);
    }, IDLE_DIM_DELAY_MS);
}

function setAvatarEmotion(emotion) {
    if (!emotion) {
        emotion = "neutral";
    }

    if (emotion === currentEmotion) {
        return;
    }

    currentEmotion = emotion;
    resetIdleDimTimer();

    const avatarImage = document.getElementById("avatar-image");
    const mergedEmotion = window.__MERGED_EMOTION__ || "";

    avatarImage.onerror = () => {
        avatarImage.classList.remove("meme-bounce");
        if (!avatarImage.src.endsWith("/neutral.png")) {
            avatarImage.src = NEUTRAL_AVATAR_SRC;
        }
    };

    avatarImage.style.opacity = "0.2";

    setTimeout(() => {
        avatarImage.src = `/static/assets/${emotion}.png`;
        avatarImage.style.opacity = "1";

        if (mergedEmotion && emotion === mergedEmotion) {
            avatarImage.classList.add("meme-bounce");
        } else {
            avatarImage.classList.remove("meme-bounce");
        }
    }, 80);
}

async function fetchEmotionState() {
    try {
        const response = await fetch("/emotion_state", {
            cache: "no-store"
        });

        const data = await response.json();
        setAvatarEmotion(data.emotion);
    } catch (error) {
        console.error("Failed to fetch emotion state:", error);
    }
}

setInterval(fetchEmotionState, 150);
fetchEmotionState();
