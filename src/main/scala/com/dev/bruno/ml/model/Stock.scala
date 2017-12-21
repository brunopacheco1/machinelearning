package com.dev.bruno.ml.model

import java.sql.Timestamp

@SerialVersionUID(100L)
class Stock extends Serializable {

  private var _name: String = _

  private var _nameIndex: Double = .0

  private var _date: Timestamp = _

  private var _open: Double = .0

  private var _close: Double = .0

  private var _low: Double = .0

  private var _high: Double = .0

  private var _volume: Double = .0

  def name: String = _name

  def name(name: String): Unit = {
    this._name = name
  }

  def nameIndex: Double = _nameIndex

  def nameIndex(nameIndex: Double): Unit = {
    this._nameIndex = nameIndex
  }

  def date: Timestamp = _date

  def date(date: Timestamp): Unit = {
    this._date = date
  }

  def open: Double = _open

  def open(open: Double): Unit = {
    this._open = open
  }

  def close: Double = _close

  def close(close: Double): Unit = {
    this._close = close
  }

  def low: Double = _low

  def low(low: Double): Unit = {
    this._low = low
  }

  def high: Double = _high

  def high(high: Double): Unit = {
    this._high = high
  }

  def volume: Double = _volume

  def volume(volume: Double): Unit = {
    this._volume = volume
  }
}