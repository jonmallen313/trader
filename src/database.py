"""
Database module for persistent trade storage.
Stores all trades ever made by users for historical analysis.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import json
import os
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def get_db_connection():
    """Get a connection to the PostgreSQL database."""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise Exception("DATABASE_URL environment variable not set")
    return psycopg2.connect(database_url)

def init_database():
    """Initialize the trades database with required tables."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                user_id TEXT DEFAULT 'default_user',
                algorithm_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                pnl REAL NOT NULL,
                pnl_percent REAL,
                order_id TEXT,
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP NOT NULL,
                duration_seconds INTEGER,
                strategy TEXT,
                timeframe TEXT,
                real_trade BOOLEAN DEFAULT FALSE,
                is_crypto BOOLEAN DEFAULT FALSE,
                metadata JSONB,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        ''')
        
        # Create algorithms table for session tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS algorithm_sessions (
                id SERIAL PRIMARY KEY,
                algorithm_id TEXT NOT NULL UNIQUE,
                user_id TEXT DEFAULT 'default_user',
                symbol TEXT NOT NULL,
                capital REAL NOT NULL,
                splits INTEGER NOT NULL,
                take_profit REAL NOT NULL,
                stop_loss REAL NOT NULL,
                strategy TEXT,
                timeframe TEXT,
                started_at TIMESTAMP NOT NULL,
                ended_at TIMESTAMP,
                status TEXT NOT NULL,
                final_pnl REAL DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                real_trading BOOLEAN DEFAULT FALSE,
                is_crypto BOOLEAN DEFAULT FALSE,
                exit_reason TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        ''')
        
        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_algo ON trades(algorithm_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_user ON trades(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_created ON trades(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user ON algorithm_sessions(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_started ON algorithm_sessions(started_at)')
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("PostgreSQL database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False


def save_trade(trade_data: dict, algorithm_id: str, user_id: str = 'default_user'):
    """
    Save a completed trade to the database.
    
    Args:
        trade_data: Dictionary containing trade information
        algorithm_id: ID of the algorithm that made the trade
        user_id: User identifier (default: 'default_user')
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Calculate additional metrics
        entry_time = trade_data.get('entry_time')
        exit_time = trade_data.get('timestamp') or datetime.now().isoformat()
        
        duration_seconds = None
        if entry_time:
            try:
                entry_dt = datetime.fromisoformat(entry_time)
                exit_dt = datetime.fromisoformat(exit_time)
                duration_seconds = int((exit_dt - entry_dt).total_seconds())
            except:
                pass
        
        pnl_percent = None
        if trade_data.get('entry_price') and trade_data.get('entry_price') > 0:
            pnl_percent = (trade_data.get('pnl', 0) / (trade_data.get('entry_price') * trade_data.get('qty', 1))) * 100
        
        # Store metadata as JSON
        metadata = {
            'position_id': trade_data.get('position_id'),
            'close_failed_count': trade_data.get('close_failed_count', 0),
            'order_details': trade_data.get('order_details', {})
        }
        
        cursor.execute('''
            INSERT INTO trades (
                user_id, algorithm_id, symbol, side, quantity,
                entry_price, exit_price, pnl, pnl_percent, order_id,
                entry_time, exit_time, duration_seconds,
                strategy, timeframe, real_trade, is_crypto, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        ''', (
            user_id,
            algorithm_id,
            trade_data.get('symbol'),
            trade_data.get('side'),
            trade_data.get('qty'),
            trade_data.get('entry_price'),
            trade_data.get('exit_price'),
            trade_data.get('pnl'),
            pnl_percent,
            trade_data.get('order_id'),
            entry_time,
            exit_time,
            duration_seconds,
            trade_data.get('strategy'),
            trade_data.get('timeframe'),
            trade_data.get('real_trade', False),
            trade_data.get('is_crypto', False),
            json.dumps(metadata)
        ))
        
        trade_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Trade saved to database: ID={trade_id}, Symbol={trade_data.get('symbol')}, P&L=${trade_data.get('pnl'):.2f}")
        return trade_id
        
    except Exception as e:
        logger.error(f"Error saving trade to database: {e}")
        return None


def save_algorithm_session(algo_data: dict, user_id: str = 'default_user'):
    """
    Save or update an algorithm session to the database.
    
    Args:
        algo_data: Dictionary containing algorithm information
        user_id: User identifier (default: 'default_user')
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if session already exists
        cursor.execute('SELECT id FROM algorithm_sessions WHERE algorithm_id = %s', (algo_data['id'],))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing session
            cursor.execute('''
                UPDATE algorithm_sessions SET
                    status = %s,
                    ended_at = %s,
                    final_pnl = %s,
                    total_trades = %s,
                    win_rate = %s,
                    exit_reason = %s
                WHERE algorithm_id = %s
            ''', (
                algo_data.get('status'),
                datetime.now() if algo_data.get('completed') else None,
                algo_data.get('pnl', 0),
                algo_data.get('total_trades', 0),
                algo_data.get('win_rate', 0),
                algo_data.get('exit_reason'),
                algo_data['id']
            ))
            logger.info(f"Algorithm session updated: {algo_data['id']}")
        else:
            # Insert new session
            cursor.execute('''
                INSERT INTO algorithm_sessions (
                    algorithm_id, user_id, symbol, capital, splits,
                    take_profit, stop_loss, strategy, timeframe,
                    started_at, status, real_trading, is_crypto
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                algo_data['id'],
                user_id,
                algo_data['symbol'],
                algo_data['capital'],
                algo_data['splits'],
                algo_data['take_profit'],
                algo_data['stop_loss'],
                algo_data.get('strategy'),
                algo_data.get('timeframe'),
                algo_data.get('started_at'),
                algo_data['status'],
                algo_data.get('real_trading', False),
                algo_data.get('is_crypto', False)
            ))
            logger.info(f"New algorithm session saved: {algo_data['id']}")
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error saving algorithm session: {e}")
        return False


def get_all_trades(user_id: str = 'default_user', limit: int = 100, offset: int = 0):
    """
    Retrieve all trades for a user.
    
    Args:
        user_id: User identifier
        limit: Maximum number of trades to return
        offset: Number of trades to skip
    
    Returns:
        List of trade dictionaries
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute('''
            SELECT * FROM trades
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        ''', (user_id, limit, offset))
        
        trades = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [dict(trade) for trade in trades]
        
    except Exception as e:
        logger.error(f"Error retrieving trades: {e}")
        return []


def get_trades_by_symbol(symbol: str, user_id: str = 'default_user', limit: int = 50):
    """Get all trades for a specific symbol."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute('''
            SELECT * FROM trades
            WHERE user_id = %s AND symbol = %s
            ORDER BY created_at DESC
            LIMIT %s
        ''', (user_id, symbol, limit))
        
        trades = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [dict(trade) for trade in trades]
        
    except Exception as e:
        logger.error(f"Error retrieving trades by symbol: {e}")
        return []


def get_algorithm_sessions(user_id: str = 'default_user', limit: int = 50):
    """Get all algorithm sessions for a user."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute('''
            SELECT * FROM algorithm_sessions
            WHERE user_id = %s
            ORDER BY started_at DESC
            LIMIT %s
        ''', (user_id, limit))
        
        sessions = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [dict(session) for session in sessions]
        
    except Exception as e:
        logger.error(f"Error retrieving algorithm sessions: {e}")
        return []


def get_trading_stats(user_id: str = 'default_user'):
    """
    Get overall trading statistics for a user.
    
    Returns:
        Dictionary with trading statistics
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Total trades
        cursor.execute('SELECT COUNT(*) FROM trades WHERE user_id = %s', (user_id,))
        total_trades = cursor.fetchone()[0]
        
        # Total P&L
        cursor.execute('SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE user_id = %s', (user_id,))
        total_pnl = cursor.fetchone()[0]
        
        # Win rate
        cursor.execute('SELECT COUNT(*) FROM trades WHERE user_id = %s AND pnl > 0', (user_id,))
        winning_trades = cursor.fetchone()[0]
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Real vs paper trades
        cursor.execute('SELECT COUNT(*) FROM trades WHERE user_id = %s AND real_trade = TRUE', (user_id,))
        real_trades = cursor.fetchone()[0]
        
        # Crypto vs stock trades
        cursor.execute('SELECT COUNT(*) FROM trades WHERE user_id = %s AND is_crypto = TRUE', (user_id,))
        crypto_trades = cursor.fetchone()[0]
        
        # Average P&L
        cursor.execute('SELECT COALESCE(AVG(pnl), 0) FROM trades WHERE user_id = %s', (user_id,))
        avg_pnl = cursor.fetchone()[0]
        
        # Best trade
        cursor.execute('SELECT COALESCE(MAX(pnl), 0) FROM trades WHERE user_id = %s', (user_id,))
        best_trade = cursor.fetchone()[0]
        
        # Worst trade
        cursor.execute('SELECT COALESCE(MIN(pnl), 0) FROM trades WHERE user_id = %s', (user_id,))
        worst_trade = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return {
            'total_trades': total_trades,
            'total_pnl': round(float(total_pnl), 2),
            'win_rate': round(win_rate, 2),
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'real_trades': real_trades,
            'paper_trades': total_trades - real_trades,
            'crypto_trades': crypto_trades,
            'stock_trades': total_trades - crypto_trades,
            'avg_pnl': round(float(avg_pnl), 2),
            'best_trade': round(float(best_trade), 2),
            'worst_trade': round(float(worst_trade), 2)
        }
        
    except Exception as e:
        logger.error(f"Error getting trading stats: {e}")
        return {}


# Initialize database on module import
init_database()
