package com.dev.bruno.ml.model

import java.sql.Timestamp

@SerialVersionUID(100L)
class Ema extends Serializable {

  private var _nameIndex: Double = .0

  private var _ema6: Double = .0

  private var _ema10: Double = .0

  private var _date: Timestamp = _

  def this(nameIndex: Double, ema6: Double, ema10: Double, date: Timestamp) {
    this()
    this._nameIndex = nameIndex
    this._date = date
    this._ema6 = ema6
    this._ema10 = ema10
  }

  def nameIndex: Double = _nameIndex

  def nameIndex(nameIndex: Double): Unit = {
    this._nameIndex = nameIndex
  }

  def date: Timestamp = _date

  def date(date: Timestamp): Unit = {
    this._date = date
  }

  def ema6: Double = _ema6

  def ema6(ema6: Double): Unit = {
    this._ema6 = ema6
  }

  def ema10: Double = _ema10

  def ema10(ema10: Double): Unit = {
    this._ema10 = ema10
  }
}