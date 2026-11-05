import sqlite3

from langchain_core.tools import tool

from src.memory.store import get_memory_store


@tool
def remember_fact(key: str, value: str) -> str:
    """ユーザーや周囲の人の事実を長期記憶に保存します。ユーザーが名前や好みなどを教えた時に使います。
    keyは「ユーザーの名前」のような短い見出し、valueは内容です。同じkeyは上書きされます。"""
    try:
        get_memory_store().remember_fact(key, value)
        return f"記憶しました: {key} = {value}"
    except sqlite3.Error as e:
        return f"記憶エラー: {str(e)}"


@tool
def forget_fact(key: str) -> str:
    """長期記憶から事実を削除します。ユーザーが忘れてほしいと言った時に使います。"""
    try:
        if get_memory_store().forget_fact(key):
            return f"忘れました: {key}"
        return f"その記憶はありません: {key}"
    except sqlite3.Error as e:
        return f"記憶エラー: {str(e)}"
