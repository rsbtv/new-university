-- ========================
-- DDL: Создание таблиц
-- ========================

-- Справочник стран
CREATE TABLE countries (
    id   SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE
);

-- Справочник жанров
CREATE TABLE genres (
    id   SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE
);

-- Пользователи
CREATE TABLE users (
    id       SERIAL PRIMARY KEY,
    name     VARCHAR(150) NOT NULL,
    email    VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL
);

-- Артисты
CREATE TABLE artists (
    id           SERIAL PRIMARY KEY,
    is_group     BOOLEAN      NOT NULL DEFAULT FALSE,  -- FALSE = соло, TRUE = коллектив
    created_date DATE,
    name         VARCHAR(200) NOT NULL,
    country_id   INT          REFERENCES countries(id) ON DELETE SET NULL
);

-- Релизы (альбомы/синглы)
CREATE TABLE releases (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    release_date DATE,
    artist_id   INT          REFERENCES artists(id) ON DELETE CASCADE
);

-- Треки
CREATE TABLE tracks (
    id         SERIAL PRIMARY KEY,
    name       VARCHAR(255) NOT NULL,
    duration   INT          NOT NULL,  -- длительность в секундах
    genre_id   INT          REFERENCES genres(id) ON DELETE SET NULL,
    artist_id  INT          REFERENCES artists(id) ON DELETE SET NULL,
    release_id INT          REFERENCES releases(id) ON DELETE SET NULL
);

-- Плейлисты
CREATE TABLE playlists (
    id           SERIAL PRIMARY KEY,
    name         VARCHAR(255) NOT NULL,
    created_date DATE         NOT NULL DEFAULT CURRENT_DATE,
    creator_id   INT          REFERENCES users(id) ON DELETE SET NULL,
    type         VARCHAR(50)  NOT NULL DEFAULT 'custom'  -- 'custom', 'album', 'chart' и т.д.
);

-- Связка пользователь ↔ плейлист (подписки/библиотека)
CREATE TABLE user_playlists (
    id          SERIAL PRIMARY KEY,
    playlist_id INT NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
    user_id     INT NOT NULL REFERENCES users(id)     ON DELETE CASCADE,
    UNIQUE (playlist_id, user_id)
);

-- Содержимое плейлиста (треки в плейлисте)
CREATE TABLE playlist_tracks (
    id          SERIAL PRIMARY KEY,
    playlist_id INT NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
    track_id    INT NOT NULL REFERENCES tracks(id)    ON DELETE CASCADE,
    UNIQUE (playlist_id, track_id)
);

-- Добавленные (лайкнутые) треки пользователя
CREATE TABLE liked_tracks (
    id      SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id)   ON DELETE CASCADE,
    track_id INT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    UNIQUE (user_id, track_id)
);

-- Любимые артисты пользователя
CREATE TABLE liked_artists (
    id        SERIAL PRIMARY KEY,
    user_id   INT NOT NULL REFERENCES users(id)    ON DELETE CASCADE,
    artist_id INT NOT NULL REFERENCES artists(id)  ON DELETE CASCADE,
    UNIQUE (user_id, artist_id)
);