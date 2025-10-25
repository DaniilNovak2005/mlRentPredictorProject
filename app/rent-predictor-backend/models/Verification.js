const mongoose = require('mongoose');

const VerificationSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        required: true,
        ref: 'User'
    },
    token: {
        type: String,
        required: true
    },
    createdAt: {
        type: Date,
        default: Date.now,
        expires: 3600 // expires in 1 hour
    },
    isForAccountCreation: {
        type: Boolean,
        required: true,
        default: true
    }
});

module.exports = mongoose.model('Verification', VerificationSchema);