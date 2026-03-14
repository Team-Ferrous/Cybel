import React, { useEffect, useState } from 'react';

const BOARD_SIZE = 12; // Square board
const REFRESH_RATE = 75; // ms - faster

// Standard Tetris pieces (tetrominos)
const TETROMINOS = [
  {
    piece: [
      [1, 1, 1, 1], // I piece
    ],
    width: 4,
    color: 'rgba(0, 240, 240, 0.25)', // Cyan
  },
  {
    piece: [
      [1, 1],
      [1, 1], // O piece (square)
    ],
    width: 2,
    color: 'rgba(240, 240, 0, 0.25)', // Yellow
  },
  {
    piece: [
      [0, 1, 0],
      [1, 1, 1], // T piece
    ],
    width: 3,
    color: 'rgba(160, 0, 240, 0.25)', // Purple
  },
  {
    piece: [
      [0, 1, 1],
      [1, 1, 0], // S piece
    ],
    width: 3,
    color: 'rgba(0, 240, 0, 0.25)', // Green
  },
  {
    piece: [
      [1, 1, 0],
      [0, 1, 1], // Z piece
    ],
    width: 3,
    color: 'rgba(240, 0, 0, 0.25)', // Red
  },
  {
    piece: [
      [1, 0, 0],
      [1, 1, 1], // J piece
    ],
    width: 3,
    color: 'rgba(0, 0, 240, 0.25)', // Blue
  },
  {
    piece: [
      [0, 0, 1],
      [1, 1, 1], // L piece
    ],
    width: 3,
    color: 'rgba(240, 160, 0, 0.25)', // Orange
  },
];

const LoadingState = ({ isConnected }) => {
  const [board, setBoard] = useState(
    Array(BOARD_SIZE)
      .fill()
      .map(() => Array(BOARD_SIZE).fill(null))
  );
  const [currentPiece, setCurrentPiece] = useState(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [sequenceIndex, setSequenceIndex] = useState(0);
  const [shouldClear, setShouldClear] = useState(false);
  const [dots, setDots] = useState('');
  const [pieceBag, setPieceBag] = useState([]);

  // Initialize or refill the piece bag
  const initializePieceBag = () => {
    const newBag = [...TETROMINOS];
    // Shuffle the bag using Fisher-Yates algorithm
    for (let i = newBag.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [newBag[i], newBag[j]] = [newBag[j], newBag[i]];
    }
    return newBag;
  };

  // Get random piece using the bag system
  const getRandomPiece = () => {
    let currentBag = [...pieceBag];

    // If bag is empty, refill it
    if (currentBag.length === 0) {
      currentBag = initializePieceBag();
      setPieceBag(currentBag);
    }

    // Take the first piece from the bag
    const pieceInfo = currentBag[0];
    // Remove it from the bag
    setPieceBag(currentBag.slice(1));

    // Calculate random starting position
    const maxX = BOARD_SIZE - pieceInfo.width;
    const startX = Math.floor(Math.random() * maxX);

    return { ...pieceInfo, startX };
  };

  // Initialize the piece bag on component mount
  useEffect(() => {
    setPieceBag(initializePieceBag());
  }, []);

  // Animate the loading dots
  useEffect(() => {
    const dotsInterval = setInterval(() => {
      setDots((prev) => (prev.length >= 3 ? '' : prev + '.'));
    }, 500);

    return () => clearInterval(dotsInterval);
  }, []);

  // Check if a piece can be placed at the given position
  const canPlace = (piece, posX, posY) => {
    for (let y = 0; y < piece.length; y++) {
      for (let x = 0; x < piece[y].length; x++) {
        if (piece[y][x]) {
          const boardY = posY + y;
          const boardX = posX + x;
          if (
            boardY >= BOARD_SIZE || // Bottom collision
            boardX < 0 || // Left wall
            boardX >= BOARD_SIZE || // Right wall
            (boardY >= 0 && board[boardY][boardX] !== null) // Piece collision
          ) {
            return false;
          }
        }
      }
    }
    return true;
  };

  // Place the current piece on the board
  const placePiece = () => {
    const newBoard = board.map((row) => [...row]);
    for (let y = 0; y < currentPiece.piece.length; y++) {
      for (let x = 0; x < currentPiece.piece[y].length; x++) {
        if (currentPiece.piece[y][x]) {
          const boardY = position.y + y;
          if (boardY >= 0) {
            newBoard[boardY][position.x + x] = currentPiece.color;
          }
        }
      }
    }
    setBoard(newBoard);
    return newBoard;
  };

  // Find the lowest possible position for the current piece
  const findLowestPosition = (piece, startX) => {
    let y = 0;
    while (canPlace(piece, startX, y + 1)) {
      y++;
    }
    return y;
  };

  // Start next piece
  const nextPiece = () => {
    const pieceInfo = getRandomPiece();
    setCurrentPiece(pieceInfo);
    setPosition({ x: pieceInfo.startX, y: 0 });
    setSequenceIndex((prev) => prev + 1);
  };

  // Game loop
  useEffect(() => {
    if (shouldClear) {
      const timer = setTimeout(() => {
        setBoard(
          Array(BOARD_SIZE)
            .fill()
            .map(() => Array(BOARD_SIZE).fill(null))
        );
        setShouldClear(false);
        nextPiece();
      }, REFRESH_RATE * 2);
      return () => clearTimeout(timer);
    }

    if (!currentPiece) {
      nextPiece();
      return;
    }

    const timer = setInterval(() => {
      const nextY = position.y + 1;

      if (canPlace(currentPiece.piece, position.x, nextY)) {
        setPosition((prev) => ({ ...prev, y: nextY }));
      } else {
        placePiece();
        // Clear board when it's getting too full (more than 8 rows filled)
        const filledRows = board.filter((row) => row.some((cell) => cell !== null)).length;
        if (filledRows > 8) {
          setShouldClear(true);
        } else {
          nextPiece();
        }
      }
    }, REFRESH_RATE);

    return () => clearInterval(timer);
  }, [currentPiece, position, sequenceIndex, shouldClear, board]);

  return (
    <div className="loading-container">
      <div className="loading-content">
        <div className="tetris-board">
          {board.map((row, y) => (
            <div key={y} className="tetris-row">
              {row.map((cell, x) => {
                let isCurrent = false;
                let currentColor = null;
                if (currentPiece) {
                  const pieceY = y - position.y;
                  const pieceX = x - position.x;
                  if (
                    pieceY >= 0 &&
                    pieceY < currentPiece.piece.length &&
                    pieceX >= 0 &&
                    pieceX < currentPiece.piece[pieceY].length
                  ) {
                    isCurrent = currentPiece.piece[pieceY][pieceX] === 1;
                    if (isCurrent) {
                      currentColor = currentPiece.color;
                    }
                  }
                }
                return (
                  <div
                    key={x}
                    className="tetris-cell"
                    style={{
                      backgroundColor: cell || currentColor || '#ffffff',
                    }}
                  />
                );
              })}
            </div>
          ))}
        </div>
        <p className="loading-text text-xs">
          {isConnected ? `Loading${dots}` : `Connecting${dots}`}
        </p>
      </div>
    </div>
  );
};

export default LoadingState;
