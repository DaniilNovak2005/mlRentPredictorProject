const mongoose = require('mongoose');

const reportSchema = new mongoose.Schema({
  token: {
    type: String,
    required: true,
    unique: true,
    index: true
  },
  data: {
    type: Object,
    default: null
  },
  status: {
    type: String,
    enum: ['pending', 'completed', 'failed'],
    default: 'pending'
  },
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

// Update the updatedAt field on save
reportSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

module.exports = mongoose.model('Report', reportSchema);