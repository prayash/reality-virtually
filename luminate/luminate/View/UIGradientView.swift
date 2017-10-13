//
//  UIGradientView.swift
//  luminate
//
//  Created by Jesse Litton on 10/7/17.
//  Copyright © 2017 com.reality.af. All rights reserved.
//

import UIKit


@IBDesignable final class GradientView: UIView {
    
    @IBInspectable var startColor: UIColor = UIColor.clear
    @IBInspectable var endColor: UIColor = UIColor.clear
    @IBInspectable var startPosition: CGPoint = CGPoint(x:0, y:0)
    @IBInspectable var endPosition: CGPoint = CGPoint(x: 1, y: 1)
    
    override func draw(_ rect: CGRect) {
        let gradient: CAGradientLayer = CAGradientLayer()
        gradient.startPoint = startPosition
        gradient.endPoint = endPosition
        gradient.frame = CGRect(x: CGFloat(0),
                                y: CGFloat(0),
                                width: superview!.frame.size.width,
                                height: superview!.frame.size.height)
        gradient.colors = [startColor.cgColor, endColor.cgColor]
        gradient.zPosition = -1
        layer.addSublayer(gradient)
    }
    
}


